#!/bin/bash
# Configures dependencies on a new DigitalOcean droplet with Dokku installed,
# adds a Git remote for deployment, and deploys the scheduling service.
# Plugins used:
#  * Redis (short-term persistence of jobs)
#  * dokku-nginx-max-upload-size (support for big POST requests)
#  * Let's Encrypt (HTTPS)

usage() {
    echo "Usage: $0 [host] [service name] [git remote name] [Let's Encrypt email]"
    exit 1;
}

setup() {
    host=$1
    name=$2
    git_remote=$3
    email=$4
    echo $host $name $git_remote $email

    deploy_user="dokku@${host}"
    d="ssh root@${host} dokku"

    echo "Navigate to ${host} and complete the setup wizard if it has not"
    echo "yet been completed. Press enter to continue."
    read

    echo "Installing plugins..."
    $d plugin:install https://github.com/dokku/dokku-redis.git
    $d plugin:install https://github.com/dokku/dokku-letsencrypt.git
    $d plugin:install https://github.com/Zeilenwerk/dokku-nginx-max-upload-size.git

    echo "Initializing app and Redis server..."
    $d apps:create $name
    $d redis:create $name
    $d redis:link $name $name

    echo "Pushing application..."
    git remote add $git_remote $deploy_user:$name
    git push -f $git_remote main:master

    echo "Configuring domain..."
    $d domains:enable $name  # enable vhost mode
    $d domains:clear $name
    $d domains:clear-global
    $d domains:add $name $host
    $d domains:set $name $host
    $d domains:set-global $host

    echo "Configuring nginx..."
    $d config:set $name MAX_UPLOAD_SIZE=20M

    echo "Configuring Let's Encrypt..."
    $d config:set --no-restart $name DOKKU_LETSENCRYPT_EMAIL=$email
    $d letsencrypt $name
    $d letsencrypt:cron-job --add
}

if [ $# -ne 4 ]
then
    usage 
else
    setup $1 $2 $3 $4
fi
