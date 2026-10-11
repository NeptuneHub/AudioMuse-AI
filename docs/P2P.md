# P2P Sharing

P2P Sharing lets AudioMuse-AI servers swap analysis results with each other, so a song that another
server has already analysed does not need the slow models again on yours.

It is **off by default**. One switch in the Setup Wizard (section **P2P Sharing**) turns on both
directions at once: your server asks others for results and answers their questions. With the switch
off there is no upload, no download and no network connection, and analysis runs exactly as it always
did.

**No open port, no public IP address and no project server are needed.** Every AudioMuse-AI server
with P2P Sharing on is a Tor onion service: other servers reach it only through the Tor network, by its
onion address, and it never accepts a connection from the internet. It works the same behind a home
router, a carrier NAT, a firewall or inside Kubernetes. The project runs no server for P2P Sharing and
never sees what is exchanged.

## How it works

1. Analysis downloads each new song and runs MusiCNN locally and, when available, Chromaprint, as
   always. They give the song its unique id and its audio fingerprint. MusiCNN is never taken from a
   peer: it is what identifies the song.
2. A song that already has every result in your catalogue gets nothing from peers. For a song that
   is new to your catalogue, the worker asks the P2P node for the other results: the DCLAP
   embedding, the GTE lyrics embedding with its axis scores, and the neural fingerprint.
3. The node sends a short audio signature (derived from the MusiCNN embedding) and the song length to
   the other AudioMuse-AI servers that currently answer (at most `P2P_MAX_PEERS`, picked at random),
   each over its own Tor connection. Results are exchanged only for the models both servers run with
   the same model files and settings. A server that has the song offers its results; your node
   confirms that the offer is the same song, fetches the results and hands them to the worker.
4. The worker saves them with the same code a local analysis uses. Whatever no peer supplied within
   `P2P_LOOKUP_TIMEOUT_SECONDS` (default 15 seconds) is computed locally.

Songs already in your catalogue that were analysed only by some of the models (for example because
you turned a model on later) ask peers for the missing ones first, and the download is skipped when
nothing else needs the audio.

The node is its own supervised process, `p2p-node`, inside the Flask container (or the native app).
Workers never talk to the network: they reach the node through PostgreSQL, so remote worker
containers need no port and no change. Only one node runs per database.

## The network: Tor onion services, no open ports

The `p2p-node` process starts its own Tor, the official prebuilt program of the Tor Project (the
Ubuntu `tor` package in the image), and drives it with stem, the Tor Project's Python controller
library. Tor and the node listen only on `127.0.0.1`, and Tor publishes the node's socket as an onion
service. As a result:

* no port is opened on the network, no public IP address is needed and there is no router setting;
* it works behind any NAT: a home router, a carrier NAT, a firewall, a Kubernetes cluster;
* everything two servers exchange travels through the Tor network, end-to-end encrypted, and neither
  server learns the other's IP address, only its onion address;
* there is no project server: the project runs nothing and never sees what is exchanged.

Tor is slower than a direct connection: a new connection to another server takes a few seconds, then
each exchange takes about one second. That is fine for these results, which are small (a few kilobytes
per song, tens of kilobytes with the neural fingerprint). When the node starts, Tor needs from a few
seconds to a few minutes to join the Tor network, and the status reads **Starting** until then.

### Finding other servers

Servers find each other in three ways:

1. **A public directory on Nostr relays.** Nostr relays are public message servers run by third
   parties. The release ships a default list (`P2P_DIRECTORY_RELAYS`, an environment variable, never
   a wizard setting) and the node reaches them through Tor. Each server publishes one signed record
   (Nostr kind 34101, `d` tag `audiomuse-ai-p2p`) that holds only its onion address, its AudioMuse-AI
   version and the ids of its models. The record is published again every 3 hours and replaces the
   previous one. The node reads the records of the last 24 hours every 10 minutes, and hourly once
   `P2P_MAX_PEERS` servers answer.
2. **Introductions.** Servers that talk to each other also list the servers they reached recently, so
   one known server leads to others.
3. **The saved list.** The node keeps the servers it met in the database (the `app_config` row
   `p2p_peers`, up to 256 servers seen in the last 7 days), so after a restart it finds them again
   without depending on the directory.

With `P2P_DIRECTORY_RELAYS` empty there is no directory and only the servers already known are used.

### Which servers answer

The node keeps no connection open. A server counts as **live** for 20 minutes after its last answer,
and the node checks it again about every 10 minutes. A server that does not answer is tried again
after 1 minute, then after a pause that doubles each time, up to 32 minutes. Songs are asked only of
live servers that run the same MusiCNN model.

## When a result from another server is used

A result from another server is used when both of these hold, and nothing else is checked:

* **Same song:** the offering server's song has the same unique id and the same Chromaprint as yours.
  It is the rule your own catalogue uses to recognise one song in two files: the audio signature,
  the song length and the MusiCNN embedding distance must match, and so must the Chromaprint when the
  Chromaprint gate (`CHROMAPRINT_GATE_ENABLED`) is on. With the gate on and a Chromaprint on your
  side, an offer without a Chromaprint is refused.
* **Version:** the result was computed by AudioMuse-AI v3.0.0 or newer (see the version rule below).

Your server then only decodes the bytes into the shape it stores: the exact size, finite numbers,
and for the neural fingerprint the same codebook as yours (another codebook cannot be decoded).
Data that does not decode is dropped and logged (error 7003) and that result is computed locally.
Nothing is ever banned, and results are not recomputed locally to double-check them, so a result
that passes these checks is used as it was sent.

**Different rips:** every lossy re-encode of the same rip (MP3, AAC, Opus, Vorbis, resampled) is
recognised as the same song. A different rip of the song (another offset, trim or gain) is
recognised only about 55 percent of the time in our measurement; when it is not, the song is simply
analysed locally.

## What is shared and what is never shared

Shared with a server that asks about a song you have:

* the DCLAP embedding, the lyrics embedding and lyrics axis scores, and the neural fingerprint;
* the MusiCNN embedding and Chromaprint of that song, so the asking server can confirm it is the same
  song;
* the AudioMuse-AI version that computed each result.

Every stored result is offered, whether your server computed it or received it from a peer. A
received result is passed on with the version of the server that computed it.

Seen by the other AudioMuse-AI servers: your onion address, your AudioMuse-AI version, a short audio
signature and the length of each song you look up, and whether your library has a song someone asks
about.

Listed in the public directory, where anyone can read it: your onion address, your AudioMuse-AI
version and the ids of the models you run (hashes of the model files and settings).

Nobody sees your IP address. Other servers reach you only through Tor, and the Nostr relays are
reached through Tor too, so they only see the public records. There is no project server. As for any
Tor user, the first Tor relay your server connects to sees that your IP address uses Tor, but not
which servers it talks to or what it exchanges.

Never sent: audio, file names or paths, titles, artists, album names, lyrics text, play history,
playlists, user accounts, passwords, media server addresses and your catalogue or media server IDs.

Also never shared: the lyrics "instrumental" marker, because it records that your server found no
lyrics source, not a fact about the song, and would hide real lyrics on the other server.

## The version rule

Every result travels with the AudioMuse-AI version that computed it. **Only results from AudioMuse-AI
v3.0.0 or newer are uploaded or accepted.** v3.0.0 is the first release after the DCLAP model change
and the switch to the GTE lyrics model, so older results live in different model spaces and would be
wrong on a newer server. Results below v3.0.0, or with a version that cannot be read, are ignored on
both sides.

Your server records the version in the database every time it saves a result, whether P2P Sharing
is on or off. Results analysed before the version was recorded carry none: they are shared with your
server's current AudioMuse-AI version.

## Privacy notice

This is the notice shown in the Setup Wizard:

> P2P Sharing is off until you turn it on. When on, this server swaps analysis results with other
> AudioMuse-AI servers through the Tor network, so new songs can skip slow analysis steps. No open
> port or public IP address is needed and no server run by the project is involved. Other servers
> never see this server's IP address: they see its onion address, its AudioMuse-AI version, a short
> audio signature and the length of each song it looks up, and whether this library has a song they
> ask about. The onion address, the AudioMuse-AI version and the models in use are listed in a public
> directory on Nostr relays. Audio, file names, titles, artists, lyrics text, play history and
> account details are never sent. Only results from AudioMuse-AI v3.0.0 or newer are exchanged, and
> only when the song matches.

Anyone can run a server and ask questions. Lyrics text is never sent, only the lyrics embedding.

## Settings

All settings live in the Setup Wizard. Saving the wizard restarts the P2P node with the new values.
See [PARAMETERS](PARAMETERS.md) for the full list and defaults.

| Setting | What it does |
|---|---|
| Switch (`P2P_ENABLED`) | Turns upload and download on or off together. |
| Parallel downloads / uploads | How many transfers run at the same time in each direction (1 to 64). A server that is full answers "busy". |
| Download / upload limit (KiB/s) | Bandwidth caps on the sharing protocol. `0` means unlimited. The directory and the checks of which servers answer are small and not counted. |
| P2P Sharing internals (Advanced) | Most servers asked about each song (`P2P_MAX_PEERS`, default 32) and the lookup timeout (`P2P_LOOKUP_TIMEOUT_SECONDS`, default 15 seconds). |

There is no network setting: no port, no address, no router mapping.

The status line in the section shows one word: **Off**, **Unavailable**, **Starting**, **Connected**,
**No peers found**, **Not running** or **Error**. It is read when you open the section or click
Refresh. The node refreshes its status every 15 seconds; a status older than a minute reads as Not
running. **Connected** means at least one other server answers. **No peers found** means none answers
yet: no other server is listed in the directory or known, or none could be reached through Tor.

## Native installs

The native apps for Linux, macOS and Windows include the P2P libraries but not Tor. To use P2P
Sharing there, install Tor first:

* **Linux:** the `tor` package of your distribution's package manager (for example
  `sudo apt install tor` or `sudo dnf install tor`).
* **macOS:** `brew install tor` with Homebrew, or the Tor Expert Bundle from the Tor Project.
* **Windows:** the Tor Expert Bundle from the Tor Project; unpack it into a folder of your choice.

The node looks for a program called `tor` on the PATH. When Tor is somewhere else, set the
environment variable `TOR_BINARY` to the full path of the program (for example
`/opt/homebrew/bin/tor` or `C:\Tor\tor\tor.exe`). `TOR_BINARY` is an environment variable only, never
a wizard setting. The node starts its own Tor process with its own ports and data folder, so a Tor
service already running on the machine is neither needed nor disturbed.

Without Tor the node reports **Unavailable** (error 7001) and the wizard refuses to turn P2P Sharing
on.

## Rejected alternatives

Direct connections between servers were rejected as the base of the network: most servers sit behind
a home router or a carrier NAT, and asking every user to forward a port or publish an address would
leave most of them out. Hole punching gets two such servers talking for only about 70 percent of
pairs, and WebRTC needs TURN relay servers for 20 to 25 percent of consumer connections. Joining the
public libp2p network and its public relays was tried first and rejected: those relays cut a relayed
connection after two minutes and 128 KiB, and its bootstrap servers stop on 2026-09-30. A main
server run by the project, relaying every exchange, was rejected because the project must not run
infrastructure for the network or know what is exchanged. Publishing one record per song in a public
index was rejected because a library has hundreds of thousands of songs. Flooding every question
through gossip was rejected for now; it is the planned path once the network grows beyond
`P2P_MAX_PEERS`.

## Not supported

The **noavx2** image includes neither Tor nor the P2P libraries. There the node reports
**Unavailable** and the wizard refuses to turn P2P Sharing on (error 7001).

## Backups and the network identity

The onion service key is stored in the database (the `app_config` row `p2p_onion_key`) the first time
the node starts and is never changed by the application, so the onion address stays the same across
restarts. The key that signs the directory record is derived from it. A backup restore brings the
key back, and with it the onion address. Two servers restored from one backup and running at the
same time share one onion address and one directory record, so the network sees them as one server.

## Error codes

| Code | Meaning |
|---|---|
| 7001 | P2P Sharing is not available: the Tor program or the P2P libraries are not installed (for example the noavx2 image, or a native install without Tor). The node reports Unavailable and stays idle, and the wizard refuses to turn it on. |
| 7002 | Peers could not supply a song in time, or the node is not running. The song is analysed locally; the message is logged for every such song. After two timeouts in a row the rest of the album stops asking. |
| 7003 | A peer's result was dropped: older than v3.0.0 or an unreadable version, bytes that do not decode, or a malformed message. The result is computed locally. |
| 7005 | The P2P node could not start or stopped unexpectedly: Tor could not start or connect to the Tor network, the stored onion key is unreadable, or the database is unreachable. It is started again automatically. |

See [ERROR_CODES](ERROR_CODES.md) for the full list.
