# Smart-Energy

Repository containing the library smart_energy, which allows the user to run simulations of energy-grid behavior.

## Getting-started

### Installing

```
git clone git@github.com:javiermas/smart-energy.git;
cd smart-energy;
pip install -e .;
pip install -r requirements.txt
```

### Running

The simulation platform requires a running instance of mongo (run the ```mongod``` command) on the default adress 127.0.0.1 and port 27017.

```
python scripts/run_environment.py
```
