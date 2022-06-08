import argparse
import numpy
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from smartarp import SimulationConfiguration, System, output_create_results, output_plot_timeseries, cost_analysis

# Main
if __name__ == "__main__":
    # Change case if specified
    parser = argparse.ArgumentParser(description="Parses the inputs for the module to run.")
    parser.add_argument('-i', '--input', dest='input', required=True, type=str, help="Input JSON file path.")
    parser.add_argument('-o', '--output', dest='output', required=True, type=str, help="Output folder.")
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', help="Plots the results.")
    parser.add_argument('-s', '--solver', dest='solver', type=str, help="Solver name. Ex: glpk, cplex, etc.",
                        default='glpk')
    args = parser.parse_args()

    numpy.random.seed(42)  # Fixing random scenario generation

    configuration = SimulationConfiguration(path_input=args.input, path_output=args.output)
    configuration.load(args)
    system = System()
    system.run(configuration)
    print("\nSystem simulated, outputing results...")
    cost_analysis(system, configuration)
    output_create_results(system, configuration)
    if args.plot:
        output_plot_timeseries(system, configuration)

    print("\nSimulation complete.")
