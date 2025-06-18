source ~/.zshrc
conda activate blip-env8

echo "injecting keplerian isotropic stoch foreground"
run_blip  params_kepinjection.ini
echo "recovering keplerian isotropic stoch foreground"
run_blip params_keprecovery.ini
echo "recovering orbiting isotropic stoch foreground"
run_blip params_orbrecovery.ini
