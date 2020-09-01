---
title: "Google Compute Engine SSH Tunnel"
date: 2020-09-01T13:03:47+02:00
draft: false
showTOC: false
---

### TLDR

You can create a ssh tunnel from your virtual machine to your local machine by running the following command on your local machine:

```bash
$ gcloud compute ssh <INSTANCE-NAME> -- \
  -N -L <LOCAL-PORT>:localhost:<REMOTE-PORT>
```

### Example

For example, if you are training a neural network using tensorflow on your virtual machine named `instance-1`. To monitor the training, you would launch `tensorboard` on the remote machine:

```bash
$ tensorboard --logdir=log
```

However `tensorboard` would only start on [localhost:6006](http://localhost:6006) and your virtual machine is usually not configured to open this port on the internet. To access this port as if the `tensorboard` was running on your machine you would run on your local machine:

```bash
$ gcloud compute ssh instance-1 -- -N -L 8080:localhost:6006
```

You could finally open [localhost:8080](http://localhost:8080) in a browser on your local machine and the tensorboard interface would show up.