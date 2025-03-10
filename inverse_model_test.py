import sys
import numpy as np
import matplotlib.pyplot as plt
from lab_amoro.parallel_robot import *
from biglide_models import *  # Modify this to test the biglide
import rclpy


# Function to initialize Matplotlib figures and axes
def init_plots():
    fig_position, axs_position = plt.subplots(1, 2, figsize=(10, 5))
    fig_velocity, axs_velocity = plt.subplots(1, 2, figsize=(10, 5))
    fig_acceleration, axs_acceleration = plt.subplots(1, 2, figsize=(10, 5))
    fig_effort, axs_effort = plt.subplots(1, 2, figsize=(10, 5))

    # Initialize position plot lines
    line_q11, = axs_position[0].plot([], [], 'm-', label='q11')
    line_q11_model, = axs_position[0].plot([], [], 'c-', label='q11 computed')
    line_q21, = axs_position[1].plot([], [], 'm-', label='q21')
    line_q21_model, = axs_position[1].plot([], [], 'c-', label='q21 computed')

    axs_position[0].set_title('Position q11 [m]')
    axs_position[1].set_title('Position q21 [m]')
    axs_position[0].set_ylim([-0.5, 0.5])
    axs_position[1].set_ylim([-0.5, 0.5])

    axs_position[0].legend()
    axs_position[1].legend()

    # Initialize velocity plot lines
    line_q11D, = axs_velocity[0].plot([], [], 'm-', label='q11D')
    line_q11D_model, = axs_velocity[0].plot([], [], 'c-', label='q11D computed')
    line_q21D, = axs_velocity[1].plot([], [], 'm-', label='q21D')
    line_q21D_model, = axs_velocity[1].plot([], [], 'c-', label='q21D computed')

    axs_velocity[0].set_title('Velocity q11 [m/s]')
    axs_velocity[1].set_title('Velocity q21 [m/s]')
    axs_velocity[0].set_ylim([-0.2, 0.2])
    axs_velocity[1].set_ylim([-0.2, 0.2])

    axs_velocity[0].legend()
    axs_velocity[1].legend()

    # Initialize acceleration plot lines
    line_q11DD, = axs_acceleration[0].plot([], [], 'm-', label='q11DD')
    line_q11DD_model, = axs_acceleration[0].plot([], [], 'c-', label='q11DD computed')
    line_q21DD, = axs_acceleration[1].plot([], [], 'm-', label='q21DD')
    line_q21DD_model, = axs_acceleration[1].plot([], [], 'c-', label='q21DD computed')

    axs_acceleration[0].set_title('Acceleration q11 [m/s²]')
    axs_acceleration[1].set_title('Acceleration q21 [m/s²]')
    axs_acceleration[0].set_ylim([-0.2, 0.2])
    axs_acceleration[1].set_ylim([-0.2, 0.2])

    axs_acceleration[0].legend()
    axs_acceleration[1].legend()

    # Initialize effort plot lines
    line_tau1, = axs_effort[0].plot([], [], 'm-', label='tau_left')
    line_tau1_model, = axs_effort[0].plot([], [], 'c-', label='tau_left computed')
    line_tau2, = axs_effort[1].plot([], [], 'm-', label='tau_right')
    line_tau2_model, = axs_effort[1].plot([], [], 'c-', label='tau_right computed')

    axs_effort[0].set_title('Effort left [N]')
    axs_effort[1].set_title('Effort right [N]')
    axs_effort[0].set_ylim([-0.05, 0.05])
    axs_effort[1].set_ylim([-0.05, 0.05])

    axs_effort[0].legend()
    axs_effort[1].legend()

    return {
        'fig_position': fig_position, 'axs_position': axs_position,
        'lines_position': [line_q11, line_q11_model, line_q21, line_q21_model],
        'fig_velocity': fig_velocity, 'axs_velocity': axs_velocity,
        'lines_velocity': [line_q11D, line_q11D_model, line_q21D, line_q21D_model],
        'fig_acceleration': fig_acceleration, 'axs_acceleration': axs_acceleration,
        'lines_acceleration': [line_q11DD, line_q11DD_model, line_q21DD, line_q21DD_model],
        'fig_effort': fig_effort, 'axs_effort': axs_effort,
        'lines_effort': [line_tau1, line_tau1_model, line_tau2, line_tau2_model]
    }


def update_plots(plot_data, time, data):
    # Update position plots
    plot_data['lines_position'][0].set_data(time, data['position'][0])  # q11
    plot_data['lines_position'][1].set_data(time, data['position'][1])  # q11_model
    plot_data['lines_position'][2].set_data(time, data['position'][2])  # q21
    plot_data['lines_position'][3].set_data(time, data['position'][3])  # q21_model

    # Update velocity plots
    plot_data['lines_velocity'][0].set_data(time, data['velocity'][0])  # q11D
    plot_data['lines_velocity'][1].set_data(time, data['velocity'][1])  # q11D_model
    plot_data['lines_velocity'][2].set_data(time, data['velocity'][2])  # q21D
    plot_data['lines_velocity'][3].set_data(time, data['velocity'][3])  # q21D_model

    # Update acceleration plots
    plot_data['lines_acceleration'][0].set_data(time, data['acceleration'][0])  # q11DD
    plot_data['lines_acceleration'][1].set_data(time, data['acceleration'][1])  # q11DD_model
    plot_data['lines_acceleration'][2].set_data(time, data['acceleration'][2])  # q21DD
    plot_data['lines_acceleration'][3].set_data(time, data['acceleration'][3])  # q21DD_model

    # Update effort plots
    plot_data['lines_effort'][0].set_data(time, data['effort'][0])  # tau1
    plot_data['lines_effort'][1].set_data(time, data['effort'][1])  # tau1_model
    plot_data['lines_effort'][2].set_data(time, data['effort'][2])  # tau2
    plot_data['lines_effort'][3].set_data(time, data['effort'][3])  # tau2_model

    # Rescale axes to fit the data
    for ax in plot_data['axs_position'].flatten():
        ax.relim()
        ax.autoscale_view()

    for ax in plot_data['axs_velocity'].flatten():
        ax.relim()
        ax.autoscale_view()

    for ax in plot_data['axs_acceleration'].flatten():
        ax.relim()
        ax.autoscale_view()

    for ax in plot_data['axs_effort'].flatten():
        ax.relim()
        ax.autoscale_view()

    # Redraw the plots
    plot_data['fig_position'].canvas.draw()
    plot_data['fig_velocity'].canvas.draw()
    plot_data['fig_acceleration'].canvas.draw()
    plot_data['fig_effort'].canvas.draw()

    plt.pause(0.001)  # Allows for dynamic updating


def main(args=None):
    # Initialize and start the ROS2 robot interface
    rclpy.init(args=args)
    robot = Robot("five_bar")  # Modify this to test the biglide
    start_robot(robot)

    # Initialize Matplotlib plots
    plot_data = init_plots()

    # Start oscillations
    robot.start_oscillate()

    time_data = []
    position_data = [[], [], [], []]  # For q11, q21 and their models
    velocity_data = [[], [], [], []]  # For q11D, q21D and their models
    acceleration_data = [[], [], [], []]  # For q11DD, q21DD and their models
    effort_data = [[], [], [], []]  # For tau1, tau2 and their models

    try:
        while True:
            if robot.data_updated():
                # Test of the Inverse Geometric model
                q11 = robot.active_left_joint.position
                q21 = robot.active_right_joint.position
                x = robot.end_effector.position_x
                y = robot.end_effector.position_y

                q11_model, q21_model = igm(x, y, -1, -1)

                # Test of the Inverse Kinematic Model
                q12 = robot.passive_left_joint.position
                q22 = robot.passive_right_joint.position
                q11D = robot.active_left_joint.velocity
                q21D = robot.active_right_joint.velocity
                xD = robot.end_effector.velocity_x
                yD = robot.end_effector.velocity_y

                q11D_model, q21D_model = ikm(q11, q12, q21, q22, xD, yD)

                # Test of the Inverse Kinematic Model Second order
                q12D = robot.passive_left_joint.velocity
                q22D = robot.passive_right_joint.velocity
                q11DD = robot.active_left_joint.acceleration
                q21DD = robot.active_right_joint.acceleration
                xDD = robot.end_effector.acceleration_x
                yDD = robot.end_effector.acceleration_y

                q11DD_model, q21DD_model = ikm2(q11, q12, q21, q22, q11D, q12D, q21D, q22D, xDD, yDD)

                # Test of the Inverse Dynamic Model
                tau1 = robot.active_left_joint.effort
                tau2 = robot.active_right_joint.effort

                M, c = dynamic_model(q11, q12, q21, q22, q11D, q12D, q21D, q22D)
                qDD = np.array([q11DD, q21DD])
                tau_model = M.dot(qDD) + c

                time = robot.get_time()

                # Append new data to lists for plotting
                time_data.append(time)
                position_data[0].append(q11)
                position_data[1].append(q11_model)
                position_data[2].append(q21)
                position_data[3].append(q21_model)

                velocity_data[0].append(q11D)
                velocity_data[1].append(q11D_model)
                velocity_data[2].append(q21D)
                velocity_data[3].append(q21D_model)

                acceleration_data[0].append(q11DD)
                acceleration_data[1].append(q11DD_model)
                acceleration_data[2].append(q21DD)
                acceleration_data[3].append(q21DD_model)

                effort_data[0].append(tau1)
                effort_data[1].append(tau_model[0])
                effort_data[2].append(tau2)
                effort_data[3].append(tau_model[1])

                # Update plots
                data = {
                    'position': position_data,
                    'velocity': velocity_data,
                    'acceleration': acceleration_data,
                    'effort': effort_data
                }
                update_plots(plot_data, time_data, data)

                robot.continue_oscillations()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv)
