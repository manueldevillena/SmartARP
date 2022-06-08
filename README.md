# SmartARP_simulator

Simulates the model of interaction of a retailer with a portfolio of consumers and flexible consumers. 
The retailer forecasts prices of electricity (day-ahead market), prices of imbalance, and demand of its consumers. 
The flexible consumers forecast its own demand which is then added to the retailer's forecast. 
Finally, the flexible consumers offer flexibility bids that the retailer may select before doing the day-ahead 
demand provisioning and before physical delivery of the electricity. The goal of using flexibility is being able to 
adapt to potential forecast errors which may lead to increased operation costs.

## Setup
This tool is written in python 3.7.
The software and its dependencies should be installed in a python virtual environment. 
First, create a virtual environment called `venv`:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

Make sure that `pip` is up-to-date:

```bash
(venv) pip install --upgrade pip
```

Install the required packages with 

```bash
pip install -r requirements
```

Then install a linear optimization solver and make sure it is available from the command-line.
By default, `glpk` is used. On Ubuntu, it may be installed with the command `sudo apt install glpk-utils`.
 
Faster simulation may be obtained using commercial solvers such as CPLEX. 
In the latter case, the solver name should be provided in the command line options.

## Running example

An example case can be run with 

```bash
python -m smartarp -i instances/example_test.json -o results -p
```

More information can be obtained, by running the help function:

```bash
python -m smartarp -h
```

A one year example is also available and can be run with CPLEX

```bash
python -m smartarp -i instances/example_long.json -o results -p -s cplex 
```
