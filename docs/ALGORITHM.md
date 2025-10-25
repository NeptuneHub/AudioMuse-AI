# Introduction

## **Table of Contents**

- [Architecture](#architecture)
- [Analysis](#analysis)

# Architecture

AudioMuse-AI is composed by this component:
- 1 x Flask container
- n Workers container
- Redis
- Postgresql

**Flask container** host the web integrated front-end an the API Server. It address all the Syncronous call and also enqueue on the Redis Qeuue all the asyncronous call. He is the main entry point for AudioMuse-AI and by default is exposed on port `8000`

**Worker** container can be one or more. Especially in multiple node deployment or in single node but with recent CPU can be deployed more than one. They address all the Asyncronous task that are mainly Analysis and Clustering. They read the task enqueued on the Redis Queue.

**Redis** is used, as said to enqueue task and, off course **Postgresql** is the database for the persistence for all the information.

From a network point of view Flask and Workers container need to be able to reach all the other component AND the Media Server BUT they can be also deployed on different machine.

# Analysis

Analysis is the first