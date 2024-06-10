eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_nbmodular
git pull $1
