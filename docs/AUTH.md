# Authentication

From v1.0.0, only PostgreSQL, Redis, and `TZ` configuration must still be configured via environment variables. All other configuration values are managed through the browser setup wizard and persisted in the database. For compatibility with legacy installations, environment variables are imported into the database automatically on first startup. The Setup Wizard is shown on clean installation as lending page and is also available later from the menu under Administration > Setup Wizard.

Authentication is enabled by default from v0.9.6 thanks to the parameters `AUTH_ENABLED`; it use this mandatory parameters  `AUDIOMUSE_USER`, `AUDIOMUSE_PASSWORD` and this optional `API_TOKEN`, `JWT_SECRET`.

The `API_TOKEN` is need only for external plugin use. The `JWT_SECRET` is if you want to keep the session when you restart the container.

The web UI provides a `/login` page where the user posts the username/password and receives a JWT cookie on success.  Subsequent browser requests are authenticated via that cookie.

Machine‑to‑machine callers may bypass the login page by supplying the `API_TOKEN` in an `Authorization: Bearer …` header. For example:

```bash
curl -v \
  -X POST 'http://192.168.3.233:8000/api/analysis/start' \
  -H 'Authorization: Bearer 123456' \
  -H 'Content-Type: application/json' \
  -d '{}'
```

# Password reset

If you have lost access to all admin accounts, reset admin access by deleting both the legacy admin config entries and the admin rows in `audiomuse_users`.

From an ubuntu cli you can install the postgresql client if you don't already have it:
```
sudo apt update && sudo apt install -y postgresql-client
```

Then replace these parameters:
- `PGPASSWORD=audiomusepassword`: database password
- `-U audiomuse`: database user
- `-d audiomusedb`: database name
- `-h 192.168.3.213`: database host
- `-p 5432`: database port

Run:
```
PGPASSWORD=audiomusepassword psql -h 192.168.3.213 -p 5432 -U audiomuse -d audiomusedb \
  -c "DELETE FROM app_config WHERE key IN ('AUDIOMUSE_USER','AUDIOMUSE_PASSWORD'); DELETE FROM audiomuse_users WHERE role = 'admin';"
```
If everything is configured correctly you should see something like:
```
DELETE 2
DELETE 1
```

Then restart the Flask and worker containers. On next access you'll be able to set a new admin user and password.

If another admin still has access, do not use this procedure; the other admin can delete and recreate the admin account from the web UI.

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

> Actually Jellyfin plugin `v0.1.51` (for Jellyfin 10.10.7) and `v0.1.52` (for jellyfin 10.11) already added this support
> 
> Navidrome plugin support it from release v7 
