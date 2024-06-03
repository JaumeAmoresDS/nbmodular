eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_nbmodular
sudo hwclock -s
git push $1
