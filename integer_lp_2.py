import numpy as np
from ortools.linear_solver import pywraplp


def task_scheduling_ilp_2(tasks, task_durations, printers, adjacency_list, verbose=False):
    # Create the solver instance
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return None

    num_tasks = len(tasks)

    # Decision variables: start times for each task
    s = {}
    for i in range(num_tasks):
        s[i] = solver.NumVar(0, solver.infinity(), f'start_time_{i}')

    # Constraints: No overlapping tasks on the same printer
    printer_task_map = {}
    for i, printer in enumerate(printers):
        if printer not in printer_task_map:
            printer_task_map[printer] = []
        printer_task_map[printer].append(i)

    for printer, task_indices in printer_task_map.items():
        if len(task_indices) == 2:
            i, j = task_indices
            # Ensure that the two tasks assigned to the same printer do not overlap
            solver.Add(s[i] + task_durations[i] <= s[j])
            solver.Add(s[j] + task_durations[j] <= s[i])

    # Constraints: No adjacent tasks on different printers at the same time
    for i, j in adjacency_list:
        # Ensure that tasks on different printers that are adjacent do not overlap
        solver.Add(s[i] + task_durations[i] <= s[j])
        solver.Add(s[j] + task_durations[j] <= s[i])

    # Objective: Minimize the latest end time (makespan)
    makespan = solver.NumVar(0, solver.infinity(), 'makespan')
    for i in range(num_tasks):
        solver.Add(s[i] + task_durations[i] <= makespan)

    solver.Minimize(makespan)

    # Solve the model
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print("No optimal solution found")
        return None

    if verbose:
        print(f'Optimal makespan: {makespan.solution_value():.3f}')
        print('Detailed Schedule:')
        for i in range(num_tasks):
            start_time = s[i].solution_value()
            end_time = start_time + task_durations[i]
            print(f'Task {i} assigned to Printer {printers[i]} starts at {start_time:.3f} and ends at {end_time:.3f}')

    return makespan.solution_value()


if __name__ == "__main__":

    # Example input (4 printers)

    tasks = [0, 1, 2, 3, 4, 5, 6, 7]
    task_durations = [1, 1, 1, 1, 5, 5, 5, 5]  # Time required for each task
    robots = [1, 2, 3, 4, 1, 2, 3, 4]  # Task assignments to robots
    connected_tasks_1 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # Tasks that share physical boundaries and cannot be done simultaneously
    connected_tasks_2 = [(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]  # Tasks that share physical boundaries and cannot be done simultaneously
    connected_tasks_3 = [(0, 1), (0, 3), (1, 2), (2, 3)]  # Tasks that share physical boundaries and cannot be done simultaneously

    makespan_1 = task_scheduling_ilp_2(tasks, task_durations, robots, connected_tasks_2, verbose=True)
    makespan_2 = task_scheduling_ilp_2(tasks, task_durations, robots, connected_tasks_3)
    print(makespan_2-makespan_1)

    # Test graph coloring hypothesis.

    # k = 100
    # T = 1
    # n = len(tasks)
    # task_lists = generate_task_lists(T, n, k)

    # lists_of_numbers = []
    # for i in range(k):
    #     task_durations = task_lists[i]
    #     makespan_4_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_1)
    #     makespan_2_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_3)
    #     print(i, makespan_2_colorable - makespan_4_colorable)
    #     if makespan_2_colorable - makespan_4_colorable > 0.0001:
    #         print(i, "False")
    #         print(makespan_2_colorable, makespan_4_colorable)
    #         print(task_durations)
    #         break
    #     if abs(makespan_2_colorable - makespan_4_colorable) < 0.0001:
    #         lists_of_numbers.append(task_durations)
    #         break

    # Example input (5 printers)

    # tasks = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # # task_durations = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5], dtype="float64")  # Time required for each task
    # robots = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]  # Task assignments to robots

    # connected_tasks_1 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]  # Tasks that share physical boundaries and cannot be done simultaneously
    # connected_tasks_2 = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]  # Tasks that share physical boundaries and cannot be done simultaneously
    # #connected_tasks_3 = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (3, 4)]

    # # Test graph coloring hypothesis.

    # k = 1000
    # T = 10
    # n = len(tasks)
    # task_lists = generate_task_lists(T, n, k)

    # for i in range(k):
    #     task_durations = task_lists[i]
    #     makespan_4_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_1)
    #     makespan_3_colorable = task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_2)
    #     print(i, makespan_3_colorable - makespan_4_colorable)
    #     if makespan_3_colorable - makespan_4_colorable > 0.0001:
    #         print(i, " False")
    #         print(makespan_3_colorable, makespan_4_colorable)
    #         print(task_durations)
    #         break

    # Run the ILP scheduling
    # task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_1)
    # task_scheduling_ilp(tasks, task_durations, robots, connected_tasks_2)