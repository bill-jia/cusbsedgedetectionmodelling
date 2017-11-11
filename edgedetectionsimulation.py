#!/usr/bin/python
import sys, os, getopt
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate

def floatrange(start,end,inter):
    output = [start]
    while output[len(output)-1] < end:
        output.append(output[len(output)-1] + inter)
    return output

class Bacteria:
    #Initiate constants
    f_light_K = 0.0017 # W/m^2 constant for f_light transfer function
    B_response_max = 298 # Miller units for maximum Beta-galactosidase activity
    B_response_min = 125 # Miller units for minimum Beta-galactosidase activity
    f_logic_c0 = 0.04
    f_logic_c1 = 0.05
    f_logic_c2 = 0.011
    f_logic_n = 1.5
    LuxRtot = 2000 # nM
    LuxR_dimer_K = 270000 # nM^3
    CI_dimer_K = 5 # nM

    def f_light(light_intensity):
        #Model ompC promoter
        B_response_range = Bacteria.B_response_max - Bacteria.B_response_min
        light_response = Bacteria.f_light_K / (Bacteria.f_light_K + light_intensity)
        return light_response * B_response_range + Bacteria.B_response_min
    
    def f_logic(AHL_conc, CI_conc):
        #Model lux-lambda promoter
        f_lux_term1 = Bacteria.LuxRtot + Bacteria.LuxR_dimer_K / (4 * AHL_conc ** 2)
        f_lux_c1 = (0.5 * (f_lux_term1 - np.sqrt(f_lux_term1 ** 2 - Bacteria.LuxRtot ** 2))) * Bacteria.f_logic_c1
        f_ci_c2 = (((CI_conc / 2) + (1 / (8 * Bacteria.CI_dimer_K)) * (1 - np.sqrt(1 + 8 * Bacteria.CI_dimer_K * CI_conc))) ** Bacteria.f_logic_n) * Bacteria.f_logic_c2
        f_logic_num = Bacteria.f_logic_c0 + f_lux_c1
        f_logic_denom = 1 + Bacteria.f_logic_c0 + f_lux_c1 + f_ci_c2 + f_lux_c1 * f_ci_c2
        return f_logic_num / f_logic_denom

class Media:
    def __init__(self):
        self.AHL_history = []
        self.CI_history = []
        self.Bgal_history = []
    def get_cur_state(self):
        a = None
        b = None
        c = None
        if len(self.AHL_history) > 0:
            a = self.AHL_history[len(self.AHL_history)-1]
        if len(self.Bgal_history) > 0:
            b = self.Bgal_history[len(self.Bgal_history)-1]
        if len(self.CI_history) > 0:
            c = self.CI_history[len(self.CI_history)-1]
        return (a,b,c)

class Simulation:
    def __init__(self,argv):
        #Output folder and graph saving interval
        self.outputfolder = datetime.now().strftime("%y%m%d-%H-%M-%S")
        self.graph_interval = 10
        try:
            opts, args = getopt.getopt(argv,"o:g:",["ofile=", "ginterval="])
        except getopt.GetoptError:
            print('edgedetectionsimulation.py -o <outputfolder> -g <graph_interval>')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-o", "--ofile"):
                self.outputfolder = arg
            elif opt in ("-g", "--ginterval"):
                self.graph_interval = int(arg)
        #Granularity constants
        self.time_granularity = 10000
        self.radius_granularity = 100
        self.angle_granularity = 100
        #Setup initial conditions
        self.max_time = 24*60*60.0
        self.k1 = 0.03 # nM/hr (convert to per second)
        self.k2 = 0.012 # hr^-1 (convert to per second)
        self.k3 = 0.8 # nM/Miller
        self.k4 = 289 # Miller units
        self.AHL_Diffusion_Coef = 1.67 * (10 ** (-7)) # cm^2/s
        self.plate_radius = 4.25 # cm
        self.time_interval = self.max_time / self.time_granularity
        #self.time_interval = 24 * 3600 * self.time_granularity
        self.radius_interval = self.plate_radius / self.radius_granularity
        self.angle_interval = 2 * np.pi / self.angle_granularity

        self.radius_h = floatrange(self.radius_interval, self.plate_radius, self.radius_interval)
        self.angle_h = floatrange(self.angle_interval, 2 * np.pi, self.angle_interval)
        self.time_h = floatrange(self.time_interval, self.max_time, self.time_interval)
        #self.time_h = floatrange(self.time_interval, 24 * 3600, self.time_interval)

        self.plate = Media()
        self.plate.AHL_history.append(np.zeros(shape=(self.radius_granularity,self.angle_granularity)))
        self.plate.CI_history.append(np.zeros(shape=(self.radius_granularity,self.angle_granularity)))
        self.plate.Bgal_history.append(np.zeros(shape=(self.radius_granularity,self.angle_granularity)))

        self.light_mask = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        for i in range(0,int(self.radius_granularity/2)):
            for j in range(0,self.angle_granularity):
                self.light_mask[i,j] = 1
        #for i in range(50,55):
        #    for j in range(int(self.radius_granularity/4),3*int(self.radius_granularity/4)):
        #        self.light_mask[j,i] = 1
        #for i in range(40,):
        #    for j in range(10,15):
        #        self.light_mask[j,i] = 1

    def dedimR(self, r):
        return r / self.plate_radius

    def dedimT(self, t):
        return t * self.AHL_Diffusion_Coef / (self.plate_radius ** 2)

    def run(self):
        count = 0
        maxzeros = int(np.log10(max(self.time_h))) + 1
        print(max(self.time_h))
        print(self.graph_interval)
        for t in self.time_h:
            #print(t)
            self.Step(t)
            count += 1
            if count % self.graph_interval == 0:
                fig = plt.figure()
                #ax = Axes3D(fig)
                rad = np.linspace(0,self.plate_radius,self.radius_granularity)
                azm = np.linspace(0,2 * np.pi,self.angle_granularity)
                r,th = np.meshgrid(rad,azm)
                z = self.plate.get_cur_state()[1]
                ax = plt.subplot(1,2,1,projection="polar")
                ax.set_title("Bgal")
                colors = plt.pcolormesh(th,r,z)
                plt.plot(azm, r, color='k', ls='none')
                plt.colorbar(colors)
                plt.grid()
                z = self.plate.get_cur_state()[0]
                ax = plt.subplot(1,2,2,projection="polar")
                ax.set_title("AHL")
                colors = plt.pcolormesh(th,r,z)
                plt.plot(azm, r, color='k', ls='none')
                plt.colorbar(colors)
                plt.grid()
                #plt.show()
                if not os.path.isdir(self.outputfolder):
                    os.mkdir(self.outputfolder)
                plt.savefig( os.path.join(self.outputfolder,str(int(t)).zfill(maxzeros) + "_" +
                    str(self.time_granularity)+".png"))
                plt.close(fig)

    def Step(self, t):
        cur_state = self.plate.get_cur_state()[0].transpose()
        new_AHL_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        new_CI_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        new_Bgal_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        #f = open("debug.txt","w")
        for i in range(0,self.radius_granularity):
            #f.write("Radius: " + str(i) + "\n")
            for j in range(0,self.angle_granularity):
                new_AHL_state[i,j] = cur_state[i,j] + self.UpdateAHL_conc(cur_state,i,j) * self.dedimT(self.time_interval)
                new_CI_state[i,j] = self.k3 * Bacteria.f_light(self.light_mask[i,j])
                new_Bgal_state[i,j] = self.k4 * Bacteria.f_logic(new_AHL_state[i,j],new_CI_state[i,j])
                #f.write("Angle " + str(j) + ": " + str(new_Bgal_state[i,j]))
        #f.close()
        self.plate.AHL_history.append(new_AHL_state.transpose())
        self.plate.CI_history.append(new_CI_state.transpose())
        self.plate.Bgal_history.append(new_Bgal_state.transpose())

    def UpdateAHL_conc(self,cur_state,i,j):
        new_state = 0
        if i == 0:
            backwardradius = i
        else:
            backwardradius = i - 1
        if i == (self.radius_granularity - 1):
            forwardradius = i
        else:
            forwardradius = i + 1
        if j == 0:
            backwardangle = self.angle_granularity - 1
        else:
            backwardangle = j - 1
        if j == (self.angle_granularity - 1):
            forwardangle = 0
        else:
            forwardangle = j + 1
        if i == 0:
            new_state += (cur_state[forwardradius,j] - cur_state[backwardradius,int(j+self.angle_granularity/2) % self.angle_granularity]) / (2 * self.dedimR(self.radius_interval)) / self.dedimR(self.radius_h[i])
            new_state += (cur_state[forwardradius,j] - 2 * cur_state[i,j] + cur_state[backwardradius,int(j+self.angle_granularity/2) % self.angle_granularity]) / (self.dedimR(self.radius_interval) ** 2)
            #new_state += (cur_state[forwardradius,j] - cur_state[backwardradius,j]) / (2 * self.radius_interval) / self.dedimR(self.radius_h[i])
            #new_state += (cur_state[forwardradius,j] - 2 * cur_state[i,j] + cur_state[backwardradius,j]) / (self.radius_interval ** 2)
        elif i == (self.radius_granularity - 1):
            new_state += (cur_state[i,j] - cur_state[i-1,j]) / self.dedimR(self.radius_interval) / self.dedimR(self.radius_h[i])
            new_state += (cur_state[i,j] - 2 * cur_state[i-1,j] + cur_state[i-2,j]) / (self.dedimR(self.radius_interval) ** 2)
            #new_state += (3 * cur_state[i,j] - 4 * cur_state[i-1,j] + cur_state[i-2,j]) / (2 * self.radius_interval) / self.dedimR(self.radius_h[i])
            #new_state += (2 * cur_state[i,j] - 5 * cur_state[i-1,j] + 4 * cur_state[i-2,j] - cur_state[i-3,j]) / (self.radius_interval ** 2)
        else:
            new_state += (cur_state[forwardradius,j] - cur_state[backwardradius,j]) / (2 * self.dedimR(self.radius_interval)) / self.dedimR(self.radius_h[i])
            new_state += (cur_state[forwardradius,j] - 2 * cur_state[i,j] + cur_state[backwardradius,j]) / (self.dedimR(self.radius_interval) ** 2)

        new_state += (cur_state[i,forwardangle] - 2 * cur_state[i,j] + cur_state[i,backwardangle]) / (self.angle_interval ** 2) / (self.dedimR(self.radius_h[i]) ** 2)
        #if (cur_state[i,forwardangle] - 2 * cur_state[i,j] + cur_state[i,backwardangle]) != 0:
        #    print("radius:" + str(i) + ", angle:" + str(j) + ", curstate:" + str(cur_state[i,j]) + ", val:" + str((cur_state[i,forwardangle] - 2 * cur_state[i,j] + cur_state[i,backwardangle])))
        new_state += self.k1 * Bacteria.f_light(self.light_mask[i,j]) - self.k2 * cur_state[i,j]

        return new_state

sim = Simulation(sys.argv[1:])
sim.run()