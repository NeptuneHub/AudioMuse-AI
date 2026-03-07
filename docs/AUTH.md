# Authentication

Authentication is off by default; the server runs in legacy (open) mode when
any of `AUDIOMUSE_USER`, `AUDIOMUSE_PASSWORD` or `API_TOKEN` are empty.

To enable auth set all three environment variables (for example in a `.env`).
If you deploy to Kubernetes, **do not place these values in a ConfigMap**; use a
Secret resource so the credentials aren’t stored in plaintext imagery or git
history:

```yaml
AUDIOMUSE_USER=alice
AUDIOMUSE_PASSWORD=secret123
API_TOKEN=foo-bar-baz
JWT_SECRET=<random-string>
```

```
AUDIOMUSE_USER=alice
AUDIOMUSE_PASSWORD=secret123
API_TOKEN=foo-bar-baz
JWT_SECRET=<random-string>
```

The web UI provides a `/login` page where the user posts the username/password
and receives a JWT cookie on success.  Subsequent browser requests are
authenticated via that cookie.

Machine‑to‑machine callers may bypass the login page by supplying the
`API_TOKEN` in an `Authorization: Bearer …` header.  For example:

```bash
curl -v \
  -X POST 'http://192.168.3.233:8000/api/analysis/start' \
  -H 'Authorization: Bearer 123456' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

## HTTPS

To have a more secure Authentication running everything over HTTPS is needed to avoid that your password go in plain text. This part is something that relay from your infrastracture and not from AudioMuse-AI itself. For example if you're deploy everything on K3S thatr come with Traefik integrated, and you have certmanager with let's encrypt, you can add an IngressRoute like this:

```
apiVersion: traefik.io/v1alpha1
kind: IngressRoute
metadata:
  name: audiomuse-ingressroute
  namespace: playlist
spec:
  entryPoints:
    - websecure
  routes:
    - match: Host(`playlist.192.168.3.169.nip.io`)
      kind: Rule
      services:
        - name: audiomuse-ai-flask-service
          port: 8000
  tls:
    certResolver: letsencrypt-production
```

## Plugin 

The Jellyfin and Navidrome support of the API TOKEN is still work in progress. Enabling the authentication layer now will block the functionality of the plugin.
