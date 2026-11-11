# Reverse proxy authentication (SSO)

AudioMuse-AI can delegate login to an identity-aware reverse proxy such as
**authentik**, Authelia or oauth2-proxy. The proxy authenticates the user and
forwards the username and group list as request headers; AudioMuse maps the
groups to its two roles. Optionally, AudioMuse forwards the same identity to
**Navidrome**, so actions such as creating a playlist happen in that user's own
Navidrome account.

Both features are **off by default**. Local login, JWT sessions and the
`API_TOKEN` keep working alongside them.

- [How it works](#how-it-works)
- [Parameters](#parameters)
- [Recognising the proxy: IP and shared secret](#recognising-the-proxy-ip-and-shared-secret)
- [Roles and groups](#roles-and-groups)
- [Setup with authentik and Caddy](#setup-with-authentik-and-caddy)
- [Acting as the Navidrome user](#acting-as-the-navidrome-user) (media server support)
- [Navidrome already uses authentik](#navidrome-already-uses-authentik)
- [Security checklist](#security-checklist)
- [Troubleshooting](#troubleshooting)

## How it works

```
Browser ──► Reverse proxy ──forward auth──► authentik
                 │  (strips client headers, adds Remote-User,
                 │   Remote-Groups and X-Proxy-Secret)
                 ▼
            AudioMuse-AI  ── checks proxy IP + secret, maps groups to role
                 │
                 │  Subsonic call, header "Remote-User: <user>",
                 │  no service credentials
                 ▼
             Navidrome   ── checks AudioMuse IP (ExtAuth.TrustedSources),
                            runs the call as <user>
```

1. The proxy asks authentik whether the browser session is valid. If not, the
   user is sent to the authentik login.
2. On success the proxy removes any `Remote-User`, `Remote-Groups` and
   `X-Proxy-Secret` headers the client sent, sets its own and forwards the
   request to AudioMuse.
3. AudioMuse accepts the identity only if every configured proxy check passes
   (peer IP and/or shared secret, see below), then derives the role from the
   groups. No local account is created; the identity is evaluated per request.
4. If Navidrome passthrough is enabled, Navidrome calls made while serving that
   request carry the username in Navidrome's external-auth header.

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `AUTH_PROXY_ENABLED` | Accept identities from a reverse proxy | `false` |
| `AUTH_PROXY_TRUSTED_SOURCES` | Comma-separated IPs/CIDRs of the proxy | *(empty)* |
| `AUTH_PROXY_SHARED_SECRET` | Secret the proxy sends with every request | *(empty)* |
| `AUTH_PROXY_SECRET_HEADER` | Header carrying the secret | `X-Proxy-Secret` |
| `AUTH_PROXY_USER_HEADER` | Header carrying the username | `Remote-User` |
| `AUTH_PROXY_GROUPS_HEADER` | Header carrying the groups (pipe- or comma-separated) | `Remote-Groups` |
| `AUTH_PROXY_ADMIN_GROUP` | Group mapped to `admin` | *(empty)* |
| `AUTH_PROXY_USER_GROUP` | Group mapped to `user`; empty = every proxy user is `user` | *(empty)* |
| `AUTH_PROXY_LOGOUT_URL` | Where *Logout* sends proxy users (IdP sign-out) | *(empty)* |
| `NAVIDROME_EXTAUTH_ENABLED` | Run Navidrome calls as the proxy user | `false` |
| `NAVIDROME_EXTAUTH_USER_HEADER` | Must equal Navidrome's `ExtAuth.UserHeader` | `Remote-User` |

`AUTH_ENABLED` must stay `true`, and the setup wizard still has to be completed
once (it creates the local admin, which remains as a break-glass account).

## Recognising the proxy: IP and shared secret

AudioMuse must be sure a request really comes from the proxy, otherwise anyone
who can reach AudioMuse could send `Remote-User: admin`. Two checks exist:

| Configured | Accepted when |
|---|---|
| Only `AUTH_PROXY_TRUSTED_SOURCES` | TCP peer is in the list |
| Only `AUTH_PROXY_SHARED_SECRET` | secret header matches |
| **Both (recommended)** | peer in the list **and** secret matches |
| Neither | never - proxy auth stays off |

- The peer is the **raw socket address**. `X-Forwarded-For` is never used for
  this check, also not when `ENABLE_PROXY_FIX=true`.
- The secret is compared in constant time. Generate a long random value, e.g.
  `openssl rand -hex 32`, and keep it out of logs and repositories.
- Use the secret alone only when the proxy IP is not stable (dynamic container
  networks). Anything that learns the secret and can reach AudioMuse can then
  impersonate every user.
- Strongest in practice: AudioMuse reachable **only** from the proxy (internal
  Docker network or firewall), plus IP and secret.

If the headers arrive but a check fails, they are ignored (one warning per peer
in the log) and the normal login applies.

## Roles and groups

| Groups sent by the proxy | AudioMuse role |
|---|---|
| contains `AUTH_PROXY_ADMIN_GROUP` | `admin` |
| contains `AUTH_PROXY_USER_GROUP` | `user` |
| neither, user group configured | 403 Forbidden |
| anything, user group empty | `user` |

Group membership is read on every request, so removing a user from a group in
authentik takes effect immediately. Proxy admins are not asked for a "current
password" on the Users page; the identity provider owns re-authentication.

The top of the sidebar shows the signed-in account: user name, role (Admin or
User) and how it signed in (`SSO - <group that granted the role>` for proxy
users, `Local account` otherwise).

Proxy users are not stored as local AudioMuse accounts. The Users page shows a
notice with the signed-in identity and role, and lists only local accounts
(for a proxy user without admin rights the list is therefore empty).

## Setup with authentik and Caddy

### authentik

1. **Directory → Groups**: create `audiomuse-admins` and `audiomuse-users` and
   add the users.
2. **Applications → Providers → Create → Proxy Provider**, mode
   *Forward auth (single application)*, external host
   `https://audiomuse.example.com`.
3. **Applications → Create** an application with this provider and bind both
   groups (Policy / Group / User Bindings) so only their members get in.
4. Add the provider to the outpost (usually the embedded outpost).

The outpost returns `X-Authentik-Username` and `X-Authentik-Groups`
(pipe-separated) to the proxy.

### Caddy

```caddyfile
audiomuse.example.com {
    # `route` keeps the directives in the written order. Without it Caddy
    # sorts them itself and would run request_header AFTER forward_auth,
    # deleting the identity headers authentik just provided.
    route {
        # authentik outpost endpoints (login callback, sign-out)
        reverse_proxy /outpost.goauthentik.io/* authentik-server:9000

        # never trust identity headers coming from the client
        request_header -Remote-User
        request_header -Remote-Groups
        request_header -X-Proxy-Secret

        forward_auth authentik-server:9000 {
            uri /outpost.goauthentik.io/auth/caddy
            copy_headers X-Authentik-Username>Remote-User X-Authentik-Groups>Remote-Groups
            trusted_proxies private_ranges
        }

        reverse_proxy audiomuse-ai-flask:8000 {
            header_up X-Proxy-Secret {env.AUDIOMUSE_PROXY_SECRET}
        }
    }
}
```

Set `AUDIOMUSE_PROXY_SECRET` in Caddy's environment to the same value as
`AUTH_PROXY_SHARED_SECRET`.

### AudioMuse-AI

```env
AUTH_ENABLED=true
AUTH_PROXY_ENABLED=true
AUTH_PROXY_TRUSTED_SOURCES=172.20.0.10          # Caddy's IP as seen by AudioMuse
AUTH_PROXY_SHARED_SECRET=<openssl rand -hex 32>
AUTH_PROXY_ADMIN_GROUP=audiomuse-admins
AUTH_PROXY_USER_GROUP=audiomuse-users
AUTH_PROXY_LOGOUT_URL=https://audiomuse.example.com/outpost.goauthentik.io/sign_out
```

### Media server plugins and API clients

Plugins (Navidrome, Jellyfin) authenticate with `API_TOKEN`. If they reach
AudioMuse **through** the proxy, forward auth would redirect them to the login
page. Either let them talk to AudioMuse directly on the internal network, or
skip `forward_auth` for requests that carry the token, e.g.:

```caddyfile
        # inside the route block, BEFORE forward_auth:
        @apitoken header_regexp Authorization ^Bearer\s
        reverse_proxy @apitoken audiomuse-ai-flask:8000
```

These requests carry no secret, so AudioMuse never treats them as proxy users.
Bearer requests are still checked against `API_TOKEN` by AudioMuse itself.

## Acting as the Navidrome user

### Media server support

Proxy login to AudioMuse itself works with every media server. Acting as the
logged-in user on the media server (identity passthrough) is media-server
specific:

| Media server | Passthrough | Notes |
|---|---|---|
| Navidrome | Supported | Uses Navidrome's external auth (`ExtAuth`), see below |
| Jellyfin | Not yet | Feasible: the admin API key can act for a `userId`, and AudioMuse already resolves usernames to ids (`resolve_user`). Needs a follow-up PR (contributions welcome) |
| Emby | Not yet | Same model as Jellyfin. Needs a follow-up PR (contributions welcome) |
| Plex | Not supported | Playlists belong to the token's account; acting for another user needs per-user tokens from plex.tv and a username mapping |
| Lyrion (LMS) | Not applicable | No user accounts; playlists are server-wide |

With every server other than Navidrome, all media-server calls keep using the
configured service account, also for proxy-authenticated users.

### Navidrome

With `NAVIDROME_EXTAUTH_ENABLED=true`, every Navidrome (Subsonic) call made
while serving a proxy-authenticated request looks like this:

```
GET http://navidrome:4533/rest/createPlaylist.view?v=1.16.1&c=AudioMuse-AI&f=json&name=...
Remote-User: chris
```

Navidrome then:

1. checks that the caller's IP (the AudioMuse **flask** container) is in
   `ND_EXTAUTH_TRUSTEDSOURCES`;
2. takes the user from the header and **creates the user automatically** if it
   does not exist yet (random password);
3. runs the call as that user - the playlist belongs to *chris*.

If Navidrome does not trust AudioMuse it ignores the header and falls back to
normal Subsonic auth. AudioMuse deliberately sends **no** service credentials in
this case, so the call fails with an authentication error instead of silently
acting as the service account.

Not affected (they keep using `NAVIDROME_USER`/`NAVIDROME_PASSWORD` or
`NAVIDROME_API_KEY`):

- background jobs: analysis, clustering, scheduled tasks;
- requests from local logins or `API_TOKEN` callers;
- calls with explicit Navidrome credentials (e.g. Sonic Fingerprint's
  `navidrome_user`), unless they name the service account itself.

If Navidrome rejects the header, creating a playlist fails with error **1108
Music Server Access Denied**, which tells the user an administrator must allow
the AudioMuse host on the media server. The AudioMuse log names the likely
cause (`ND_EXTAUTH_TRUSTEDSOURCES` missing the AudioMuse IP, or a header-name
mismatch).

### Playlist visibility

Playlists created by the service account stay **public** in Navidrome, as
before, so every user can see them. Playlists created while acting as a proxy
user belong to that user and are **private by default**:

- every playlist form shows a **Share with all users** checkbox next to the
  playlist name while the passthrough is active (unticked = private); the
  checkboxes on one page share one setting;
- API callers share a playlist with `?playlist_public=true` or
  `"public": true` in the JSON body.

## Navidrome already uses authentik

If Navidrome is already behind authentik with external auth, most of the work
is done. Check these points:

1. **Add AudioMuse to Navidrome's trusted sources** - in addition to the proxy:
   ```env
   ND_EXTAUTH_TRUSTEDSOURCES=172.20.0.10/32,172.20.0.21/32   # Caddy, AudioMuse flask
   ```
   Use exact `/32` addresses, not whole subnets. Navidrome has no shared
   secret, so IP plus network isolation is the only protection on this hop.
2. **Same header name** - `NAVIDROME_EXTAUTH_USER_HEADER` must equal Navidrome's
   `ND_EXTAUTH_USERHEADER` (default `Remote-User`). If your Navidrome setup uses
   e.g. `X-Authentik-Username`, set the same value in AudioMuse.
3. **Call Navidrome directly** - `NAVIDROME_URL` must point to Navidrome on the
   internal network (`http://navidrome:4533`), **not** to the public URL behind
   the proxy. Through the proxy the header would be stripped, Navidrome would see
   the proxy's IP, and `forward_auth` would answer with a login redirect.
4. **Same usernames** - AudioMuse forwards the authentik username unchanged.
   Since Navidrome already gets its users from the same authentik, they match.
5. **Keep a working service account** - background jobs still log in with
   `NAVIDROME_USER`/`NAVIDROME_PASSWORD` or `NAVIDROME_API_KEY`. Users
   auto-created by external auth only have a random password, so use an account
   whose password you set in Navidrome (or an API key). The Subsonic login with
   these credentials keeps working because AudioMuse does not send the header
   for those calls.
6. **`/rest` exceptions in the proxy do not matter here** - AudioMuse talks to
   Navidrome directly, so rules you have for Subsonic clients stay unchanged.
7. **Trust is broad** - Navidrome will accept *any* username from AudioMuse.
   Only enable the passthrough when AudioMuse and Navidrome run in a network you
   control, and protect AudioMuse with the proxy checks above.

### Check your proxy bypass paths first

A common Navidrome setup lets `/rest/*` (Subsonic clients) and `/share/*`
bypass forward auth. If the proxy forwards those requests **without removing
the user header**, anyone on the internet can send e.g.
`X-Authentik-Username: admin` and Navidrome accepts it, because the request
comes from the trusted proxy IP. This is independent of AudioMuse, but enabling
the passthrough means trusting this header, so verify it before you start.

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

## Security checklist

- [ ] Proxy strips client-supplied `Remote-User`, `Remote-Groups` and the secret header.
- [ ] `AUTH_PROXY_TRUSTED_SOURCES` contains only the proxy (no `0.0.0.0/0`).
- [ ] `AUTH_PROXY_SHARED_SECRET` is long and random and only known to proxy and AudioMuse.
- [ ] AudioMuse's port is not published to the LAN/internet, only to the proxy.
- [ ] `ND_EXTAUTH_TRUSTEDSOURCES` lists exact IPs (proxy, AudioMuse), no broad ranges.
- [ ] Navidrome bypass paths (`/rest/*`, `/share/*`) strip the user header (test above).
- [ ] `AUTH_PROXY_LOGOUT_URL` set, so *Logout* also ends the authentik session.
- [ ] A local admin exists for when the IdP is down.

State-changing requests (POST/PUT/PATCH/DELETE) of proxy users must be
same-origin (`Sec-Fetch-Site`/`Origin` check), because the IdP session cookie is
also sent on cross-site requests.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Login page instead of SSO | Headers ignored: peer not in `AUTH_PROXY_TRUSTED_SOURCES` or secret wrong/missing (see the warning in the AudioMuse log) |
| 403 "not allowed to use AudioMuse-AI" | User in neither group, or group names differ (check `X-Authentik-Groups`) |
| Everyone is `user`, nobody `admin` | `AUTH_PROXY_ADMIN_GROUP` empty or misspelt |
| 403 "Cross-site request rejected" | Page opened via a different host name than the API calls, or a browser extension stripping `Origin` |
| Error 1108 "Music Server Access Denied" on playlist creation | AudioMuse IP missing in `ND_EXTAUTH_TRUSTEDSOURCES`, header name mismatch, or `NAVIDROME_URL` points to the proxy |
| Playlist created under the service account | `NAVIDROME_EXTAUTH_ENABLED` is off, or the action runs as a background job |
| Logout logs straight back in | `AUTH_PROXY_LOGOUT_URL` not set |
