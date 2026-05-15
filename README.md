# CPE515 Final Project
Jack Marshall and Ronnie Sidhu

## Instructions
Clone this repo to the proj directory of CFU Playground in WSL.

I added these to my .bashrc file in WSL. You gotta run them for the cfu stuff to work:
```bash
runcfu() {
    cd ~/CFU-Playground || return
    source env/conda/bin/activate cfu-symbiflow
    cd ~/CFU-Playground/proj/cpe515finalproj || return
}
exitcfu() {
	conda deactivate
	cd
}
```

To run the simulation, type:
```bash
make sim
```
Go to the menu, click "f" for full sweep.
Copy and paste whatever results you get into the /analysis/log.txt file.
Make sure you have python installed.

If you don't already have matplotlib, do:
```bash
pip install matplotlib
```
Then do:
```bash
make analyze
```

Then go to the analysis folder to see the results graphed.

