from shiny.express import input, render, ui
from shiny import reactive
import numpy as np
import matplotlib.pyplot as plt
from simulation import create_master_simulator

ui.page_opts(fillable = True)

with ui.navset_pill(id = 'tab'):

    with ui.nav_panel("Instructions"):
        with ui.layout_columns():
            with ui.card():
                ui.card_header("Theory")
                ui.markdown('''
                Scenario of patients waiting outside OPDs, dispensaries etc. can be depicted as a queue with new
                entities entering into the queue and older entities exiting out of the queue. Each entity is 
                characterised by a **waiting time** before he/she exits out of the queue. The queue is characterised 
                by varying **lengths** at given time points. The process will be characterised by **total number of 
                entities**.

                The **average waiting time**, **maximum queue length** and **total number of entities** are 
                operationally important parameters of a queue, which denotes the efficiency of the underlying 
                process and also sets the requirements for seating arrangements and number of exit points. The above 
                are important hospital performance indices.

                It is extremely difficult and operationally impractical to follow each entity with his/her 
                start time and end time to calculate the average waiting time.

                By using the **Queue Theory**, we can estimate the behaviour of the queue with just two 
                parameters: **rate of arrival into queue (entities per unit time)** and **rate of exit out of 
                queue (entities per unit time)**.
                ''')

            with ui.card():
                ui.card_header("Assumptions")

                ui.markdown('''
                1. The arrival into and exit of entities out of the queue follow **Poisson distribution** with a given 
                rate.

                2. The queue is organized one, in that the entity with earlier arrival exits earlier (in 
                a First In First Out fashion).

                3. The Poisson distribution implies that the arrivals/exits occur maximum of one at a 
                infinitesimal time interval and that the arrivals/exits are independent of earlier values.

                4. The simulation will run till the time the queue is not empty.
                ''')

            with ui.card():
                ui.card_header("Steps to conduct the survey")

                ui.markdown('''
                1. Identify the OPD to be studied.

                1. Identify the point of entry (say, registration counter of the OPD) into the queue and 
                point of exits (say, outside doctors' chambers) out of the queue.

                1. Place an observer at the point of entry and at each point of exits.

                1. Note down the start time of the OPD and number of entities already in the queue.

                1. Note down the last time of observation.

                1. Divide the total time duration into equally spaced intervals (say, intervals of 
                15 minutes each).

                1. Note the number of entities arriving into the queue in each time interval and number 
                of entities exiting out of the queue in each time interval.

                1. In the `Enter Details` tab, enter the name of department, date, width of time interval,
                unit of time interval and number of time intervals.

                1. Click on the `Create arrivals/exits` button and proceed to the next tab `Enter Arrivals and Exits`.

                1. Enter the number of arrivals and exits as measured earlier for each time intervals.

                1. If there are patients waiting in the queue before the time of start of data collection, then enter 
                the number into first `arrival` with corresponding `exit` to be zero.
                ''')

    with ui.nav_panel("Enter Details"):
        with ui.layout_columns():
            with ui.card():
                ui.card_header("Basic Details")
                ui.input_text("dept", "Department Name")
                ui.input_date("date_entry", "Date")
                ui.input_numeric("dur", "Width of each Duration", min = 1, value = 15)
                ui.input_text("dur_unit", "Unit of Duration", "minutes")
                ui.input_numeric("num_intvl", "Number of Durations", min = 1, value = 3)
                ui.input_action_button("b_create_arr_exit", "Create arrivals/exits")

    with ui.nav_panel("Enter Arrivals and Exits"):
        with ui.layout_columns():
            with ui.card():
                ui.card_header("Number of Arrivals in each Duration")

                @render.express
                @reactive.event(input.b_create_arr_exit)
                def arrivals():
                    for i in range(input.num_intvl()):
                        ui.input_numeric("arr" + str(i), None, value = 0, min = 0)
            
            with ui.card():
                ui.card_header("Number of Exits in each Duration")

                @render.express
                @reactive.event(input.b_create_arr_exit)
                def exits():
                    for i in range(input.num_intvl()):
                        ui.input_numeric("exit" + str(i), None, value = 0, min = 0)

            with ui.card():
                ui.card_header("Simulation Parameters")
                ui.input_numeric("n_exits", "Number of Exits", 1, min = 1)
                ui.input_numeric("rnd_seed", "Random number seed", 100, min = 0)
                ui.input_numeric("n_sim", "Number of simulation", 200, min = 1)
                ui.input_numeric("thresh_rate", "Threshold Rate", 0.05, min = 0.01)

        ui.input_action_button("b_sim", "Simulate")

    with ui.nav_panel("Get Report"):

        @reactive.calc
        def durs():
            return [input.dur()] * input.num_intvl()

        @reactive.calc
        def arrs():
            l_arr = []
            for i in range(input.num_intvl()):
                l_arr.append(input['arr'+str(i)]())
            return l_arr

        @reactive.calc
        def exits():
            l_exit = []
            for i in range(input.num_intvl()):
                l_exit.append(input['exit'+str(i)]())
            return l_exit

        @reactive.calc
        @reactive.event(input.b_sim)
        def sim():
            with ui.Progress(min = 0, max = 15) as p:
                np.random.seed(input.rnd_seed())
                p.set(0, "Calculating ...")
                s = create_master_simulator(arrs(), exits(), durs(), input.n_exits(), input.thresh_rate(), input.n_sim())
                p.set(15, "Completed!!")
            return s

        with ui.layout_columns():
            with ui.card():
                @render.express
                def heading():
                    ui.card_header(f"Department of {input.dept()}, \
                        {input.date_entry().strftime('%d %b %Y')}")
                
                with ui.layout_columns():
                    with ui.card():

                        @render.express
                        def waiting_time():
                            wt = sim().average_waiting_time(with_ci = True)
                            
                            with ui.value_box(theme = "blue"):
                                "Average Waiting Time"
                                
                                f"{wt.mean:.2f} {input.dur_unit()}"
                                
                                f"95% CI: {wt.cl_95[0]:.2f} - {wt.cl_95[1]:.2f} {input.dur_unit()}"
                    
                    with ui.card():
                    
                        @render.express
                        def queue_length():
                            ql = sim().max_queue_length(with_ci = True)

                            with ui.value_box(theme = "red"):
                                "Maximum Queue Length"
                                
                                f"{ql.max:3d}"
                                
                                f"95% CI: {ql.cl_95[0]:.2f} - {ql.cl_95[1]:.2f}"

                    with ui.card():

                        @render.express
                        def num_entities():
                            ne = sim().describe_num_entities()

                            with ui.value_box(theme = "green"):
                                "Average Number of Entities"

                                f"{ne.mean:.2f}"

                                f"95% CI: {ne.cl_95[0]:.2f} - {ne.cl_95[1]:.2f}"
                                

                with ui.layout_columns():

                    with ui.card():

                        ui.card_header("Average Waiting Time")

                        @render.plot
                        def plot_waiting_time():
                            f, ax = plt.subplots()
                            sim().plot_average_waiting_time_with_time(ax = ax, with_ci = False)
                            ax.set_xlabel(f'Time since start of observation ({input.dur_unit()})')
                            ax.set_ylabel(f'Mean Waiting Time ({input.dur_unit()})')

                    with ui.card():

                        ui.card_header("Distribution of Waiting Time")

                        @render.plot
                        def plot_dist_waiting_time():
                            f, ax = plt.subplots()
                            sim().get_waiting_times()
                            wts = []
                            for t in sim().waiting_times:
                                wts.append(sim().waiting_times[t])
                            wts = [i for l in wts for i in l]
                            ax.hist(wts)
                            ax.set_xlabel(f"Waiting Time ({input.dur_unit()})")
                            ax.set_ylabel("Count")

                with ui.layout_columns():
                    
                    with ui.card():
                        
                        ui.card_header("Maximum Queue Length")

                        @render.plot
                        def plot_queue_length():
                            f, ax = plt.subplots()
                            sim().plot_max_queuelength_with_time(ax, with_ci=False)
                            ax.set_xlabel(f'Time since start of observation ({input.dur_unit()})')
                            ax.set_ylabel('Maximum Queue Length')