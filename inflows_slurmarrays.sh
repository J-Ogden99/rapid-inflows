#!/bin/bash --login

conda activate inflows

ls /home/rchales/compute/era5_hourly_ro_yearly_nc/*.nc | \
slurm-auto-array --mem 12 -ntasks 2 --time 06:00:00 --mail-type=BEGIN --mail-type=END --mail-type=FAIL --mail-user=rchales@byu.edu \
-- python /home/rchales/rapid-inflows/inflows_fast.py --inflowsroot /home/rchales/compute/inflows --inputsroot /home/rchales/compute/inputs --lsmfile