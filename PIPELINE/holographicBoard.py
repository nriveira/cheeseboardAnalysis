# Code for interpretting cheeseboard coordinates and adjusting to allow for interface with ExperimentStruct
# Also allow for the ability to save metadata to a csv file to assist with fugure analysis
# by also enabling analyzing multiple videos 

# Data to include in CSV: Coordinates (normal and pixel values), startbox boundary, board adjustment values

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class HolographicBoard:
    def __init__(self, exp_tag, background_image, data_file=None):
        self.coords = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        self.x_lines = np.array([132, 165, 195, 230, 265, 300, 335, 370, 405, 440, 475, 510, 545, 580, 612])
        self.y_lines = np.array([42, 75, 107, 142, 178, 212, 250, 285, 318, 355, 390, 423, 457, 490, 523])
        self.startbox_boundary = 113
        self.trial_ID = exp_tag
        self.data_file = data_file
        self.background_image = background_image
        self.load_data(data_file, exp_tag)
        if self.R1_xcoord is not None:
            self.set_coordinates()

    def load_data(self, data_file, exp_tag):
        # Open the data file, if this is empty, create a new structure to hold the data
        if data_file is not None:
            self.data_df = pd.read_csv(data_file)
        else:
            self.data_df = pd.DataFrame(columns=['trial_ID',
                                                 'R1_xcoord', 'R1_ycoord', 'R1_xpixel', 'R1_ypixel',
                                                 'R2_xcoord', 'R2_ycoord', 'R2_xpixel', 'R2_ypixel',
                                                 'R3_xcoord', 'R3_ycoord', 'R3_xpixel', 'R3_ypixel',
                                                 'x_adj', 'y_adj',
                                                 'startbox_boundary'])
            # Save the empty dataframe to a new CSV file
            self.data_df.to_csv('board_calibration.csv', index=False)
        # Once the data is established, search the currrnet dataframe for the experimentTag
        if exp_tag in self.data_df['trial_ID'].values:
            # If the experimentTag is found, load the corresponding data into the class attributes
            exp_data = self.data_df[self.data_df['trial_ID'] == exp_tag].iloc[0]
            self.R1_xcoord = exp_data['R1_xcoord']
            self.R1_ycoord = exp_data['R1_ycoord']
            self.R2_xcoord = exp_data['R2_xcoord']
            self.R2_ycoord = exp_data['R2_ycoord']
            self.R3_xcoord = exp_data['R3_xcoord']
            self.R3_ycoord = exp_data['R3_ycoord']
            self.x_adj = exp_data['x_adj']
            self.y_adj = exp_data['y_adj']
            self.startbox_boundary = exp_data['startbox_boundary']

        else:
            # If the experimentTag is not found, initialize the attributes with default values
            self.R1_xcoord = None
            self.R1_ycoord = None
            self.R1_xpixel = None
            self.R1_ypixel = None
            self.R2_xcoord = None
            self.R2_ycoord = None
            self.R2_xpixel = None
            self.R2_ypixel = None
            self.R3_xcoord = None
            self.R3_ycoord = None
            self.R3_xpixel = None
            self.R3_ypixel = None
            self.x_adj = 0
            self.y_adj = 0
            self.startbox_boundary = 100
    
    def save_data(self, exp_tag):
        # Change the row with the exp_ID to the current values of the class attributes, if the exp_ID is not found, add a new row with the current values
        if exp_tag in self.data_df['trial_ID'].values:
            self.data_df.loc[self.data_df['trial_ID'] == exp_tag, ['R1_xcoord', 'R1_ycoord', 'R1_xpixel', 'R1_ypixel',
                                                                   'R2_xcoord', 'R2_ycoord', 'R2_xpixel', 'R2_ypixel',
                                                                   'R3_xcoord', 'R3_ycoord', 'R3_xpixel', 'R3_ypixel',
                                                                   'x_adj', 'y_adj',
                                                                   'startbox_boundary']] = [self.R1_xcoord, self.R1_ycoord, self.R1_xpixel, self.R1_ypixel,
                                                                                           self.R2_xcoord, self.R2_ycoord, self.R2_xpixel, self.R2_ypixel,
                                                                                           self.R3_xcoord, self.R3_ycoord, self.R3_xpixel, self.R3_ypixel,
                                                                                           self.x_adj, self.y_adj,
                                                                                           self.startbox_boundary]
        else:
            new_row = {'trial_ID': self.trial_ID,
                       'R1_xcoord': self.R1_xcoord, 'R1_ycoord': self.R1_ycoord, 'R1_xpixel': self.R1_xpixel, 'R1_ypixel': self.R1_ypixel, 
                       'R2_xcoord': self.R2_xcoord, 'R2_ycoord': self.R2_ycoord, 'R2_xpixel': self.R2_xpixel, 'R2_ypixel': self.R2_ypixel, 
                       'R3_xcoord': self.R3_xcoord, 'R3_ycoord': self.R3_ycoord, 'R3_xpixel': self.R3_xpixel, 'R3_ypixel': self.R3_ypixel, 
                       'x_adj': self.x_adj, 'y_adj': self.y_adj,
                       'startbox_boundary': self.startbox_boundary}
            self.data_df = pd.concat([self.data_df, pd.DataFrame([new_row])], ignore_index=True)
        self.data_df.to_csv(self.data_file, index=False)
        

    def show_board(self, lines=True):
        plt.imshow(self.background_image)
        if lines:
            # Plot lines for the grid for alignment
            for x in self.x_dict.values():
                plt.vlines(x, 0, 540, color='blue', linestyle='--')
            for y in self.y_dict.values():
                plt.hlines(y, 0, 720, color='blue', linestyle='--')

            # Plot x_dict[0] and y_dict[0] for reference
            plt.scatter(self.x_dict[0], self.y_dict[0], color='red', label='Origin (0,0)')

            # Plot the startbox boundary
            plt.vlines(self.startbox_boundary, 0, 540, color='green', linestyle='--', label='Startbox Boundary')

            # Plot reward locations if they exist
            if self.R1_xcoord is not None and self.R1_ycoord is not None:
                plt.scatter(self.R1_xpixel, self.R1_ypixel, color='red', s=100, label='Reward 1')
            if self.R2_xcoord is not None and self.R2_ycoord is not None:
                plt.scatter(self.R2_xpixel, self.R2_ypixel, color='green', s=100, label='Reward 2')
            if self.R3_xcoord is not None and self.R3_ycoord is not None:
                plt.scatter(self.R3_xpixel, self.R3_ypixel, color='blue', s=100, label='Reward 3')
            plt.legend()

    def set_coordinates(self):
        x_dict = dict(zip(self.coords, self.x_lines+self.x_adj))
        y_dict = dict(zip(self.coords, self.y_lines+self.y_adj))
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.R1_xpixel = self.x_dict[self.R1_xcoord]
        self.R1_ypixel = self.y_dict[-self.R1_ycoord]
        self.R2_xpixel = self.x_dict[self.R2_xcoord]
        self.R2_ypixel = self.y_dict[-self.R2_ycoord]
        self.R3_xpixel = self.x_dict[self.R3_xcoord]
        self.R3_ypixel = self.y_dict[-self.R3_ycoord]

    def set_reward_coordinates(self, R1_coord, R2_coord, R3_coord):
        # Flip the y coords to account for the inverted y-axis in the image
        self.R1_xcoord = R1_coord[0]
        self.R1_ycoord = R1_coord[1]
        self.R1_xpixel = self.x_dict[self.R1_xcoord]
        self.R1_ypixel = self.y_dict[-self.R1_ycoord]
        self.R2_xcoord = R2_coord[0]
        self.R2_ycoord = R2_coord[1]
        self.R2_xpixel = self.x_dict[self.R2_xcoord]
        self.R2_ypixel = self.y_dict[-self.R2_ycoord]
        self.R3_xcoord = R3_coord[0]
        self.R3_ycoord = R3_coord[1]
        self.R3_xpixel = self.x_dict[self.R3_xcoord]
        self.R3_ypixel = self.y_dict[-self.R3_ycoord]
        self.show_board()
        self.save_data(self.trial_ID)
    
    def adjust_coordinates(self, x, y):
        # Adjust coordinates to be relative to the center of the board
        self.x_adj += x
        self.y_adj += y
        x_dict = dict(zip(self.coords, self.x_lines+self.x_adj))
        y_dict = dict(zip(self.coords, self.y_lines+self.y_adj))
        self.x_dict = x_dict
        self.y_dict = y_dict
        self.show_board()
        self.save_data(self.trial_ID)

    def adjust_startbox_boundary(self, adjust):
        self.startbox_boundary = self.startbox_boundary + adjust
        self.show_board()
        self.save_data(self.trial_ID)