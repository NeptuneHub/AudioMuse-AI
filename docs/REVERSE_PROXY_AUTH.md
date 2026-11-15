# Reverse proxy authentication (SSO)

AudioMuse-AI can delegate browser login to an identity-aware reverse proxy such
as **authentik**, Authelia or oauth2-proxy. The proxy authenticates the user and
forwards the username and group list as request headers; AudioMuse maps the
groups to its two roles. Optionally, playlists that these users create can be
stored in their own **Navidrome** account instead of the service account.

Everything here is **off by default** and configured from the **environment
only**. Local login, JWT sessions and the `API_TOKEN` keep working alongside it.

- [How it works](#how-it-works)
- [Parameters](#parameters)
- [Recognising the proxy: shared secret](#recognising-the-proxy-shared-secret)
- [Roles, accounts and logout](#roles-accounts-and-logout)
- [Proxy setup](#proxy-setup) (authentik, Caddy, Traefik, clients and plugins)
- [Playlist owner for proxy users](#playlist-owner-for-proxy-users) (Navidrome)
- [Navidrome already uses authentik](#navidrome-already-uses-authentik)
- [Plugins and clients](#plugins-and-clients)
- [Security checklist](#security-checklist)
- [Troubleshooting](#troubleshooting)

## How it works

```
Browser ──► Reverse proxy ──forward auth──► authentik
                 │  strips client copies of the identity and secret
                 │  headers, then adds Remote-User, Remote-Groups and
                 │  X-Proxy-Secret (forward-auth route only)
                 ▼
            AudioMuse-AI  ── checks the shared secret (and the proxy IP, if
                             configured) and maps the groups to a role
                 │
                 │  only when a proxy user creates a playlist and
                 │  AUTH_PROXY_PLAYLIST_OWNER=user:
                 │  Subsonic call with "Remote-User: <user>", no service credentials
                 ▼
             Navidrome   ── checks the AudioMuse IP (ExtAuth.TrustedSources)
                            and creates the playlist as <user>
```

1. The proxy asks authentik whether the browser session is valid. If not, the
   user is sent to the authentik login.
2. On success the proxy removes any identity or secret headers the client sent,
   sets its own and forwards the request to AudioMuse.
3. AudioMuse accepts the identity only when the request carries the shared
   secret (and, if `AUTH_PROXY_TRUSTED_SOURCES` is set, comes from one of those
   addresses). It derives the role from the groups. No local account is
   created; the identity is evaluated on every request.
4. Everything else runs exactly as for any other session. Only the playlist
   write can optionally run as the user on Navidrome (see below).

## Parameters

All parameters are read from the environment only. They are never stored in
the database (`app_config`), never shown or editable in the setup wizard, never
returned by the API and therefore never part of a backup. Rows that earlier
test builds of this feature wrote to `app_config` are removed at startup.

| Parameter | Description | Default |
|---|---|---|
| `AUTH_PROXY_ENABLED` | Accept identities from a reverse proxy | `false` |
| `AUTH_PROXY_SHARED_SECRET` | **Required.** Secret the proxy adds on its forward-auth route. Without it proxy login stays off | *(empty)* |
| `AUTH_PROXY_SECRET_HEADER` | Header carrying the secret | `X-Proxy-Secret` |
| `AUTH_PROXY_TRUSTED_SOURCES` | Optional extra check: comma-separated IPs/CIDRs the proxy connects from | *(empty)* |
| `AUTH_PROXY_USER_HEADER` | Header carrying the username | `Remote-User` |
| `AUTH_PROXY_GROUPS_HEADER` | Header carrying the groups (pipe- or comma-separated) | `Remote-Groups` |
| `AUTH_PROXY_ADMIN_GROUP` | Group mapped to `admin` | *(empty)* |
| `AUTH_PROXY_USER_GROUP` | Group mapped to `user`; empty = every proxy user is `user` | *(empty)* |
| `AUTH_PROXY_LOGOUT_URL` | Where *Logout* sends proxy users (IdP sign-out). Local sessions always return to the login page | *(empty)* |
| `AUTH_PROXY_PLAYLIST_OWNER` | Who owns playlists that proxy users create: `service` (service account) or `user` (their own Navidrome account) | `service` |
| `NAVIDROME_EXTAUTH_USER_HEADER` | For `AUTH_PROXY_PLAYLIST_OWNER=user`: must equal Navidrome's `ExtAuth.UserHeader` | `Remote-User` |

`AUTH_ENABLED` must stay `true`, and the setup wizard still has to be completed
once (it creates the local admin, which remains as a break-glass account).

## Recognising the proxy: shared secret

AudioMuse must be sure the identity headers were set by the proxy after a
successful login; otherwise anyone could send `Remote-User: admin`.

- **The shared secret is mandatory.** The proxy adds it only on the route that
  runs forward auth. Bypass routes on the same proxy (Subsonic, plugins, apps)
  come from the same IP, so the IP alone cannot tell them apart; the secret can.
  If `AUTH_PROXY_ENABLED=true` but no secret is set, AudioMuse logs an error at
  startup and ignores all identity headers.
- `AUTH_PROXY_TRUSTED_SOURCES` is an **optional** second check. When it is set,
  the request must also come from one of those addresses. The address is the
  raw socket peer; `X-Forwarded-For` is never used, also not with
  `ENABLE_PROXY_FIX=true`.
- The secret is compared in constant time. Generate it with
  `openssl rand -hex 32`, use a different one for every service, and keep it out
  of logs and repositories.
- Strongest in practice: AudioMuse reachable **only** from the proxy (internal
  Docker network or firewall), plus secret and trusted sources.

If identity headers arrive without a valid secret, they are ignored and the
normal login applies. The log warns once per sender address (capped, so a flood
of addresses cannot fill the log or memory).

## Roles, accounts and logout

| Groups sent by the proxy | AudioMuse role |
|---|---|
| contains `AUTH_PROXY_ADMIN_GROUP` | `admin` (also when the user group is present too) |
| contains `AUTH_PROXY_USER_GROUP` | `user` |
| neither, user group configured | 403 Forbidden |
| anything, user group empty | `user` |

- Group membership is read on every request, so removing a user from a group in
  the IdP takes effect immediately.
- **A proxy identity is never a local account.** A local AudioMuse account with
  the same name is a separate account: a proxy user does not see it in the
  Users list, cannot change its password, and a proxy admin may manage it like
  any other local account. Changing a password never gives a proxy user a local
  session.
- Proxy admins are not asked for a "current password" on the Users page; the
  IdP owns re-authentication.
- The top of the sidebar shows the signed-in account: name, role and
  `SSO - <group that granted the role>` or `Local account`. The Users page tells
  proxy users that their account is managed by the IdP.
- **Logout** depends on how the user came in: a request through the proxy with
  a valid identity goes to `AUTH_PROXY_LOGOUT_URL`, so the IdP session ends
  too; a local session (for example the break-glass admin on the direct port)
  goes back to the AudioMuse login page.
- **Break-glass access:** on a route with forward auth, the proxy identity
  wins. A user who is in neither group gets a 403 there even with a valid local
  session. Use the direct port or an internal hostname without forward auth for
  the local admin.

## Proxy setup

### authentik

1. **Directory → Groups**: create `audiomuse-admins` and `audiomuse-users` and
   add the users.
2. **Applications → Providers → Create → Proxy Provider**, mode
   *Forward auth (single application)*, external host
   `https://audiomuse.example.com`.
3. **Applications → Create** an application with this provider and bind both
   groups (Policy / Group / User Bindings) so only their members get in.
4. Add the provider to the outpost (usually the embedded outpost).
5. Do not let users change their own username in authentik: the username is
   the only link to AudioMuse (and to Navidrome for `AUTH_PROXY_PLAYLIST_OWNER=user`).

The outpost returns `X-Authentik-Username` and `X-Authentik-Groups`
(pipe-separated) to the proxy.

### AudioMuse-AI

```env
AUTH_ENABLED=true
AUTH_PROXY_ENABLED=true
AUTH_PROXY_SHARED_SECRET=<openssl rand -hex 32>
AUTH_PROXY_TRUSTED_SOURCES=172.20.0.10          # optional: the proxy's IP as seen by AudioMuse
AUTH_PROXY_ADMIN_GROUP=audiomuse-admins
AUTH_PROXY_USER_GROUP=audiomuse-users
AUTH_PROXY_LOGOUT_URL=https://audiomuse.example.com/outpost.goauthentik.io/sign_out
AUTH_PROXY_PLAYLIST_OWNER=service               # or: user (see below)
```

The examples below rename the authentik headers to the defaults `Remote-User`
and `Remote-Groups`. You can also keep the authentik names and set
`AUTH_PROXY_USER_HEADER=X-Authentik-Username` and
`AUTH_PROXY_GROUPS_HEADER=X-Authentik-Groups`.

### Caddy

```caddyfile
audiomuse.example.com {
    # `route` keeps the directives in the written order. Without it Caddy
    # sorts them itself and runs request_header AFTER forward_auth, deleting
    # the identity headers authentik just provided.
    route {
        # authentik outpost endpoints (login callback, sign-out)
        reverse_proxy /outpost.goauthentik.io/* authentik-server:9000

        # never trust identity or secret headers coming from the client
        request_header -Remote-User
        request_header -Remote-Groups
        request_header -X-Authentik-Username
        request_header -X-Authentik-Groups
        request_header -X-Proxy-Secret

        forward_auth authentik-server:9000 {
            uri /outpost.goauthentik.io/auth/caddy
            copy_headers X-Authentik-Username>Remote-User X-Authentik-Groups>Remote-Groups
            trusted_proxies private_ranges
        }

        # the secret is added only here, after forward auth succeeded
        reverse_proxy audiomuse-ai-flask:8000 {
            header_up X-Proxy-Secret {env.AUDIOMUSE_PROXY_SECRET}
        }
    }
}
```

Set `AUDIOMUSE_PROXY_SECRET` in Caddy's environment to the same value as
`AUTH_PROXY_SHARED_SECRET`. Do not put the secret on any other route.

### Traefik (sketch, untested)

```yaml
# AudioMuse: AUTH_PROXY_USER_HEADER=X-authentik-username
#            AUTH_PROXY_GROUPS_HEADER=X-authentik-groups
# Services, entryPoints and TLS are left out.
http:
  middlewares:
    authentik:
      forwardAuth:
        address: http://authentik-server:9000/outpost.goauthentik.io/auth/traefik
        trustForwardHeader: true
        authResponseHeaders: [X-authentik-username, X-authentik-groups]
    audiomuse-secret:
      headers:
        customRequestHeaders:
          X-Proxy-Secret: "<same value as AUTH_PROXY_SHARED_SECRET>"
    strip-identity:
      headers:
        customRequestHeaders:   # an empty value removes the header
          X-authentik-username: ""
          X-authentik-groups: ""
          X-Proxy-Secret: ""
  routers:
    audiomuse:                  # browsers, SSO; the secret is added after forward auth
      rule: Host(`audiomuse.example.com`)
      middlewares: [authentik, audiomuse-secret]
      service: audiomuse
    audiomuse-outpost:          # authentik's sign-in callback on the app host;
                                # its longer rule gives it priority over the router above
      rule: Host(`audiomuse.example.com`) && PathPrefix(`/outpost.goauthentik.io/`)
      service: authentik
    audiomuse-clients:          # bypass for apps and plugins: AudioMuse login or API_TOKEN
      rule: Host(`audiomuse-api.example.com`)
      middlewares: [strip-identity]
      service: audiomuse
```

Traefik replaces the `X-authentik-*` headers only on routers that run
forwardAuth, which is exactly why the secret is required: it is added on the
SSO router only, and stripped on every other one.

### Clients, plugins and other bypasses

Native apps and media-server plugins cannot pass forward auth (they get a login
redirect). Give them a way in that **cannot carry an identity**:

- **Recommended:** a separate hostname or internal URL without forward auth,
  ideally reachable only from the LAN or VPN. Strip the identity and secret
  headers there and never add the secret. Clients sign in with a local account
  or use `API_TOKEN`; plugins usually talk to AudioMuse internally anyway.

  ```caddyfile
  audiomuse-api.example.com {
      @outside not remote_ip private_ranges
      respond @outside 403          # optional: LAN/VPN only
      request_header -Remote-User
      request_header -Remote-Groups
      request_header -X-Authentik-Username
      request_header -X-Authentik-Groups
      request_header -X-Proxy-Secret
      reverse_proxy audiomuse-ai-flask:8000
  }
  ```

- **If you must bypass on the SSO hostname** (for bearer-token clients only),
  exclude the login endpoints. Otherwise anyone can skip authentik by sending
  any `Authorization: Bearer ...` header and reach the local password login,
  and AudioMuse has **no limit on failed login attempts**:

  ```caddyfile
          # inside the route block, BEFORE forward_auth:
          @apitoken {
              header_regexp Authorization ^Bearer\s
              not path /auth /login
          }
          reverse_proxy @apitoken audiomuse-ai-flask:8000
  ```

  These requests carry no secret, so AudioMuse never treats them as proxy
  users, and bearer tokens are still checked against `API_TOKEN`. Clients that
  sign in with a username and password are not covered by this matcher; use a
  separate hostname for them.

- Whatever you expose without forward auth, use a long random password for the
  local admin.

## Playlist owner for proxy users

`AUTH_PROXY_PLAYLIST_OWNER` decides who owns playlists that **proxy users**
create in AudioMuse. It changes nothing else: every other media-server call
(library reads, connection tests in the setup wizard and the server registry,
plugin helpers), local sessions, `API_TOKEN` callers and background jobs
(analysis, clustering, cron) always use the service account.

| Value | Playlists of proxy users | Visibility |
|---|---|---|
| `service` (default) | created by the service account, like for everyone else | public, as before |
| `user` | created in the proxy user's own Navidrome account | private by default |

### Media server support

Proxy login itself works with every media server. `AUTH_PROXY_PLAYLIST_OWNER=user`
is media-server specific:

| Media server | `user` owner | Notes |
|---|---|---|
| Navidrome | Supported | Uses Navidrome's external auth (`ExtAuth`), see below |
| Jellyfin | Not yet | Feasible: the admin API key can act for a `userId`, and AudioMuse already resolves usernames to ids (`resolve_user`). Needs a follow-up PR (contributions welcome) |
| Emby | Not yet | Same model as Jellyfin. Needs a follow-up PR (contributions welcome) |
| Plex | Not supported | Playlists belong to the token's account; acting for another user needs per-user tokens from plex.tv and a username mapping |
| Lyrion (LMS) | Not applicable | No user accounts; playlists are server-wide |

On every server other than Navidrome, playlists keep using the service account.

### Navidrome

With `AUTH_PROXY_PLAYLIST_OWNER=user`, only the calls that create a proxy
user's playlist look like this:

```
GET http://navidrome:4533/rest/createPlaylist.view?v=1.16.1&c=AudioMuse-AI&f=json&name=...
Remote-User: chris
```

Navidrome then:

1. checks that the caller's IP (the AudioMuse **flask** container) is in
   `ND_EXTAUTH_TRUSTEDSOURCES`;
2. takes the user from the header and **creates the user automatically** if it
   does not exist yet (random password);
3. creates the playlist as that user.

AudioMuse deliberately sends **no** service credentials with these calls. If
Navidrome does not trust AudioMuse, nothing is created, nothing falls back to
the service account, and the user gets error **1108 Music Server Access
Denied**; the AudioMuse log names the likely cause (`ND_EXTAUTH_TRUSTEDSOURCES`
missing the AudioMuse IP, or a header-name mismatch).

Explicit Navidrome credentials passed with a request (another user name or
another API key, e.g. Sonic Fingerprint's `navidrome_user`) always win.

### Playlist visibility

- Service-account playlists stay **public**, so every user can see them.
- Playlists created in a proxy user's own account are **private by default**.
  Every playlist form then shows a **Share with all users** checkbox next to
  the playlist name (unticked = private); the checkboxes on one page share one
  setting. API callers share with `?playlist_public=true` or `"public": true`
  in the JSON body.

## Navidrome already uses authentik

Only relevant for `AUTH_PROXY_PLAYLIST_OWNER=user`. Check these points:

1. **Add AudioMuse to Navidrome's trusted sources** - in addition to the proxy:
   ```env
   ND_EXTAUTH_TRUSTEDSOURCES=172.20.0.10/32,172.20.0.21/32   # proxy, AudioMuse flask
   ```
   Use exact `/32` addresses, not whole subnets. Navidrome has no shared
   secret, so IP plus network isolation is the only protection on this hop.
2. **Same header name** - `NAVIDROME_EXTAUTH_USER_HEADER` must equal Navidrome's
   `ND_EXTAUTH_USERHEADER` (default `Remote-User`). If your Navidrome setup uses
   e.g. `X-Authentik-Username`, set the same value in AudioMuse.
3. **Call Navidrome directly** - `NAVIDROME_URL` must point to Navidrome on the
   internal network (`http://navidrome:4533`), **not** to the public URL behind
   the proxy. Through the proxy the header would be stripped, Navidrome would see
   the proxy's IP, and forward auth would answer with a login redirect.
4. **Same usernames** - AudioMuse forwards the IdP username unchanged. If
   Navidrome gets its users from the same IdP, they match.
5. **Keep a working service account** - everything except proxy users'
   playlists logs in with `NAVIDROME_USER`/`NAVIDROME_PASSWORD` or
   `NAVIDROME_API_KEY`. Users auto-created by external auth only have a random
   password, so use an account whose password you set in Navidrome (or an API key).
6. **Trust is broad** - Navidrome accepts *any* username, admins included,
   from the AudioMuse IP without a password. That includes code that runs in
   the AudioMuse process, such as plugins. Only use `user` when AudioMuse and
   Navidrome run in a network you control, protect AudioMuse with the proxy
   checks above, and do not let users rename themselves in the IdP.

### Check your proxy bypass paths first

A common Navidrome setup lets `/rest/*` (Subsonic clients) and `/share/*`
bypass forward auth. If the proxy forwards those requests **without removing
the user header**, anyone on the internet can send e.g.
`X-Authentik-Username: admin` and Navidrome accepts it, because the request
comes from the trusted proxy IP. This is independent of AudioMuse, but
`AUTH_PROXY_PLAYLIST_OWNER=user` means trusting this header, so verify it first.

Test through the public URL, from outside your network (or with
`curl --connect-to` to the proxy's internal address and port):

```sh
curl -H "X-Authentik-Username: someuser" \
  "https://navidrome.example.com/rest/ping.view?v=1.16.1&c=test&f=json"
```

- `"status":"ok"` - **vulnerable**: the header is honoured without a login.
- error `10` (*missing parameter 'u'*) - the header is ignored, as it should be.

Use the header name from your `ND_EXTAUTH_USERHEADER`. Fix it by removing the
header in every handler that skips forward auth:

```caddyfile
handle /rest/* {
    reverse_proxy navidrome:4533 {
        header_up -X-Authentik-Username
    }
}
handle /share/* {
    reverse_proxy navidrome:4533 {
        header_up -X-Authentik-Username
    }
}
```

Navidrome auto-creates unknown users, so a successful test leaves a new user
behind; delete it afterwards.

## Plugins and clients

- Proxy users get `g.auth_method == 'proxy'`, with `g.auth_user` set to the IdP
  username and `g.auth_role` to `admin` or `user`. See the identity contract in
  [PLUGIN.md](PLUGIN.md#request-identity). Plugins that only accept `session`
  or `bearer` must add `proxy` to serve SSO users.
- Proxy login covers AudioMuse-AI's own login only. Plugins with their own
  authentication (their own tokens, accounts or login pages) are not covered:
  proxy users are not signed in to them automatically, and the proxy login
  does not replace or protect that plugin auth.
- Plugin data keyed on `g.auth_user` treats the proxy user `alice` and a local
  account `alice` as the same principal. AudioMuse itself keeps them apart (see
  [Roles, accounts and logout](#roles-accounts-and-logout)); plugins that need
  the distinction can check `g.auth_method`.
- Existing clients keep working unchanged: `POST /auth` sessions and
  `API_TOKEN` requests behave exactly as before. SSO users cannot use native
  clients through forward auth without extra work (an in-app browser that keeps
  the proxy's session cookie, or IdP-specific app credentials). Give clients
  the separate hostname described above.

## Security checklist

- [ ] `AUTH_PROXY_SHARED_SECRET` is set, long, random, unique to AudioMuse and
      added by the proxy **only** on the forward-auth route.
- [ ] Every route to AudioMuse strips client-supplied identity headers and the
      secret header, including bypass routes.
- [ ] Clients and plugins use a separate hostname or internal URL; a bearer
      bypass on the SSO hostname excludes `/auth` and `/login`.
- [ ] Optional: `AUTH_PROXY_TRUSTED_SOURCES` contains only the proxy.
- [ ] AudioMuse's port is not published to the LAN/internet, only to the proxy.
- [ ] For `AUTH_PROXY_PLAYLIST_OWNER=user`: `ND_EXTAUTH_TRUSTEDSOURCES` lists
      exact IPs, and Navidrome bypass paths strip the user header (test above).
- [ ] `AUTH_PROXY_LOGOUT_URL` set, so *Logout* also ends the IdP session.
- [ ] A local admin with a strong password exists for when the IdP is down.

State-changing requests (POST/PUT/PATCH/DELETE) of proxy users must be
same-origin (`Sec-Fetch-Site`/`Origin` check), because the IdP session cookie is
also sent on cross-site requests.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Login page instead of SSO | Headers ignored: secret missing or wrong, or sender not in `AUTH_PROXY_TRUSTED_SOURCES` (see the warning in the AudioMuse log) |
| Log: "AUTH_PROXY_SHARED_SECRET is empty" | Proxy login is off until a secret is configured |
| Changing an `AUTH_PROXY_*` value has no effect | They are read from the environment at startup: restart the containers |
| 403 "not allowed to use AudioMuse-AI" | User in neither group, or group names differ (check the groups header) |
| Everyone is `user`, nobody `admin` | `AUTH_PROXY_ADMIN_GROUP` empty or misspelt |
| 403 "Cross-site request rejected" | Page opened via a different host name than the API calls, or a browser extension stripping `Origin` |
| Error 1108 "Music Server Access Denied" on playlist creation | AudioMuse IP missing in `ND_EXTAUTH_TRUSTEDSOURCES`, header name mismatch, or `NAVIDROME_URL` points to the proxy |
| Proxy user's playlist created under the service account | `AUTH_PROXY_PLAYLIST_OWNER` is `service` (the default), the media server is not Navidrome, or explicit credentials were passed |
| Logout logs straight back in | `AUTH_PROXY_LOGOUT_URL` not set |
| Local admin on the direct port lands on the IdP sign-out | Not expected any more: local sessions always return to the login page |
