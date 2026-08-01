# Campus Server External Access

This document records the current external access path for the campus server service exposed through the VPS.

## Current Route

- Public entry: `http://39.102.209.27:1597`
- Reverse proxy host: `39.102.209.27`
- Reverse proxy service: `nginx`
- Upstream campus server: `http://115.190.90.101:1597`

Current request flow:

`client -> 39.102.209.27:1597 -> nginx reverse proxy -> 115.190.90.101:1597 -> response -> client`

## Current Nginx Config

The active VPS config lives at `/etc/nginx/conf.d/campus-1597.conf` and proxies traffic from port `1597` to the campus server.

The route currently uses Nginx directly. The earlier SSH reverse tunnel for `1597` has already been removed.

## Verification

The route was verified from the VPS side with:

- direct upstream access to `http://115.190.90.101:1597`
- local proxy access to `http://127.0.0.1:1597`

Both returned `404` on `/`, which is expected because the upstream root path currently returns `404`. This confirms that the proxy path is working and that the root endpoint itself is simply not defined.

## Operational Notes

- If external clients cannot reach `39.102.209.27:1597`, first check the VPS security group or firewall for `1597/TCP`.
- If the upstream campus service changes port, update `/etc/nginx/conf.d/campus-1597.conf` on the VPS and reload Nginx.
- The current setup assumes the VPS can continue reaching the campus server over the campus VPN.
