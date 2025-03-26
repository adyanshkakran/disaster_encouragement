# Global imports
import os
import time
from numpy import array as vector
from numpy import matrix as matrix

# Local imports
from data_reader import Data_Reader # reader for old data format
#from data_reader_beta_matrices import Data_Reader
from seiiird_model import SEIIIRD_Model
from seiiird_tracing_model import SEIIIRD_Tracing_Model
from explicit_euler import Explicit_Euler
from result_analyzer import Result_Analyzer
from visualizer import Visualizer

def beta(t):
    if t < 100:
        return matrix(vector([((0.1)*(100 - t)/100) + 0.05]).reshape((1, 1)))
    elif t < 300:
        return matrix(vector([0.05]).reshape((1, 1)))
    elif t < 400:
        return matrix(vector([0.25 + ((0.1)*(400 - t)/100)]).reshape((1, 1)))
    else:
        return matrix(vector([0.075]).reshape((1, 1)))

def run_simulation(data_set_name):
    t_start = 0
    t_end = 600
    stepsize = 1e-2

    # Preparing data parsing
    # data_set_name = "scenarios-underlying-figure-4"
    data_directory_name = "../data/" + data_set_name + "/"
    data_directory = os.fsencode(data_directory_name)

    for file in os.listdir(data_directory):
        data_filename = os.fsdecode(file)
        if data_filename.endswith(".csv"):

            # Parsing the data
            data_filename_prefix = data_filename.split(".")[0]
            data_reader = Data_Reader()
            packed_data = data_reader.read_from_csv_file(data_directory_name + data_filename_prefix + ".csv")

            # Unpacking some of the parsed data
            tracing_data_given = packed_data[0]
            assert(isinstance(tracing_data_given, bool))
            K = packed_data[1]
            N = packed_data[2]
            beds = packed_data[-2]
            x0 = packed_data[-1]

            # Instantiate the ODE system class
            if tracing_data_given:
                print("Setting up the tracing model")
                ode_system = SEIIIRD_Tracing_Model(packed_data[1:-1])
            else:
                print("Setting up the tracing-free model")
                ode_system = SEIIIRD_Model(packed_data[1:-1], time_dependent_params={'beta_asym': beta, 'beta_sym': beta, 'beta_sev': beta})
                # ode_system = SEIIIRD_Model(packed_data[1:-1])

            # Solve the ODE system
            explicit_euler = Explicit_Euler(ode_system, stepsize)
            start_time = time.time()
            results = explicit_euler.solve(t_start, x0, t_end)
            end_time = time.time()
            print("Required CPU time = " + str(round(end_time - start_time, 3)) + " seconds")

            # Analyze results
            path = "../results/" + data_set_name + "/"
            if not os.path.exists(path):
                os.makedirs(path)
            result_analyzer = Result_Analyzer(tracing_data_given, results, stepsize,
                                            N, K, beds, t_start, t_end,
                                            path + data_filename_prefix)
            result_analyzer.extract_results()
            result_dict = result_analyzer.get_extracted_results_as_dict()
            result_analyzer.compute_and_write_results()

            # Visualize results
            visualizer = Visualizer(tracing_data_given, result_dict, N, K, beds, t_start, t_end,
                                    "../results/" + data_set_name + "/" + data_filename_prefix)

            visualizer.plot_all_curves()
            visualizer.plot_aggregated_curves()
            # visualizer.paper_plot_figure_1()
            if tracing_data_given:
                visualizer.paper_plot_figure_3()
                visualizer.paper_plot_figure_4()

def main():
    run_simulation("Alpha")
    
if __name__ == "__main__":
    main()