#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 15:21:37 2019

@author: nrchilku
"""
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt

def main(width, height, samples, frames, save=False):
    
    X = []
    Y = []
    P = []
    V = []
    screen = pygame.display.set_mode((width, height))
    end = False       
    for sample_count in range(samples):
        radius = np.random.randint(5, 20)
        x = np.random.randint(radius + 5, width - radius - 5) # initial x-position
        y = np.random.randint(radius + 5, height - radius - 5) # initial y-position
        [v_x, v_y] = np.random.rand(2)*20-10 # initial x-velocity
        color = np.random.randint(50, 255, 3) # background is black so starting at 50
        X_frames = []
        P_frames =[]
        V_frames = []
                            
        for frame_count in range(frames):                
            x += v_x
            y += v_y
            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, color, (x,y), radius)
            X_frames.append(pygame.surfarray.array3d(screen))
            P_frames.append([x,y])
            V_frames.append(v_x,v_y])
            # check for bounce
            if x <= radius or x >= width - radius:
                v_x *= -1
            if y <= radius or y >= height - radius:
                v_y *= -1  
            #clock.tick(30)
            pygame.display.flip()
        # append X and Y
        X.append(np.array(X_frames))
        # Y
        grayscale = np.dot(np.array(X_frames), np.array([0.2125, 0.7154, 0.0721]))
        Y.append(grayscale.astype('uint8'))
        # V
        V.append(V_frames)
        P.append(P_frames)
                
    
    if save:
        np.save("X_width_" + str(width) + "_" + str(frames) + "_" + str(samples), np.array(X))
        np.save("Y_width_" + str(width) + "_" + str(frames) + "_" + str(samples), np.array(Y))
        np.save("P_width_" + str(width) + "_" + str(frames) + "_" + str(samples), np.array(P))
        np.save("V_width_" + str(width) + "_" + str(frames) + "_" + str(samples), np.array(V))
        
    #return np.array(X), np.array(Y)
    pygame.quit()
                    
                
if __name__ == '__main__':
    main(224,224,10,15,save=True)
