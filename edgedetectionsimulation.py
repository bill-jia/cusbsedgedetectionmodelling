#!/usr/bin/python
import sys, os, getopt, time
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
    def __init__(self, radius_granularity, angle_granularity):
        # Store current state
        self.AHL_history = np.zeros(shape=(radius_granularity, angle_granularity))
        self.CI_history = np.zeros(shape=(radius_granularity, angle_granularity))
        self.Bgal_history = np.zeros(shape=(radius_granularity, angle_granularity))
        self.dudt = np.zeros(shape=(radius_granularity, angle_granularity))
        self.dudr = np.zeros(shape=(radius_granularity, angle_granularity))
        self.du2dr2 = np.zeros(shape=(radius_granularity, angle_granularity))
        self.du2dtheta2 = np.zeros(shape=(radius_granularity,angle_granularity))
        self.light_term = np.zeros(shape=(radius_granularity,angle_granularity))
        self.decay_term = np.zeros(shape=(radius_granularity,angle_granularity))
    def get_cur_ahl_state(self):
        return self.AHL_history
    def get_cur_state(self):
        return (self.AHL_history,self.Bgal_history,self.CI_history)
    def get_cur_derivatives(self):
        return (self.dudt, self.dudr, self.du2dr2, self.du2dtheta2, self.light_term, self.decay_term)

class Simulation:
    def __init__(self,argv):
        #Initialize adjustable parameters
        self.outputfolder = datetime.now().strftime("%y%m%d-%H-%M-%S") #Default output folder is just a timestamp
        self.graph_interval = 10
        self.plots = False
        self.max_time = 36*60*60.0
        self.plot_derivatives = False

        #Change adjustable parameters according to command line input
        try:
            opts, args = getopt.getopt(argv,"o:g:t:pd",["ofile=", "ginterval=", "plot", "time=", "derivativesplot"])
        except getopt.GetoptError:
            print('edgedetectionsimulation.py -p <makeplots> -o <outputfolder> -g <graph_interval> -t -d')
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-o", "--ofile"): #Set output folder
                self.outputfolder = arg
            elif opt in ("-p", "--plot"): #Flag to create graphs
                self.plots = True                
            elif opt in ("-g", "--ginterval"): #Set time interval at which graphs are produced
                self.graph_interval = int(arg)
            elif opt in ("-d", "derivativesplot"): # Flag to create graphs of each term in diffusion equation
                self.plot_derivatives = True
            elif opt in ("-t", "time="):
                self.max_time = float(arg)*60*60 #Set max time of simulation
        #Granularity constants
        self.time_granularity = 3000 # Time step twice as fine as space step
        self.radius_granularity = 1000
        self.angle_granularity = 100
        
        #Setup initial conditions
        self.AHL_Diffusion_Coef = 1.67 * (10 ** (-7)) # cm^2/s
        self.plate_radius = 4.25 # cm
        self.k1 = 0.03 / 3600 / (self.AHL_Diffusion_Coef/self.plate_radius**2) # nM/hr converted to nM/dimensionless time
        self.k2 = 0.012 / 3600 / (self.AHL_Diffusion_Coef/self.plate_radius**2) # hr^-1 converted to 1/dimensionless time
        self.k3 = 0.8 # nM/Miller
        self.k4 = 289.0 # Miller units
        self.time_interval = self.max_time / self.time_granularity
        self.radius_interval = self.plate_radius / self.radius_granularity
        self.angle_interval = 2.0 * np.pi / self.angle_granularity

        #Generate sampling points
        self.radius_h = floatrange(self.radius_interval, self.plate_radius, self.radius_interval)
        self.angle_h = floatrange(self.angle_interval, 2 * np.pi, self.angle_interval)
        self.time_h = floatrange(self.time_interval, self.max_time, self.time_interval)
        #self.time_h = floatrange(self.time_interval, 24 * 3600, self.time_interval)

        #Initialize plate
        self.plate = Media(self.radius_granularity, self.angle_granularity)

        #Set light input
        self.light_mask = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        for i in range(0,int(self.radius_granularity/2)):
            for j in range(0,self.angle_granularity):
                self.light_mask[i,j] = 1

        self.total_time = 0

    def dedimR(self, r):
        return r / self.plate_radius

    def dedimT(self, t):
        return t * self.AHL_Diffusion_Coef / (self.plate_radius ** 2)

    def run(self):
        count = 0

        print("Simulating to: " + str(max(self.time_h)/60) + " minutes")
        if self.plots:
            print("Plotting every: " + str(self.graph_interval) + " timepoints")
        else:
            print("No plots being made")
        print("Each time step: " + str(self.time_interval/60) + " minutes")
        
        for t in self.time_h:
            #print(t)
            start = time.time()
            self.Step(t)
            count += 1
            if count % self.graph_interval == 0 and self.plots :
                self.make_plots(t);
            end = time.time()
            print("Last step took " + str(end-start) + " seconds", end="\r")
        print(self.total_time)

    def make_plots(self,t):
        #Plot Bgal and AHL
        maxzeros = int(np.log10(max(self.time_h))) + 1
        cur_state = self.plate.get_cur_state()
        fig = plt.figure()
        fig.suptitle(str(int(t/60)) + " min")
        #ax = Axes3D(fig)
        rad = np.linspace(0,self.plate_radius,self.radius_granularity)
        azm = np.linspace(0,2 * np.pi,self.angle_granularity)
        th,r = np.meshgrid(azm,rad)
        z = cur_state[1]
        ax = plt.subplot(1,2,1,projection="polar")
        ax.set_title("Bgal")
        colors = plt.pcolormesh(th,r,z)
        plt.colorbar(colors)
        plt.grid()
        z = cur_state[0]
        ax = plt.subplot(1,2,2,projection="polar")
        ax.set_title("AHL")
        colors = plt.pcolormesh(th,r,z)
        plt.colorbar(colors)
        plt.grid()
        #plt.show()
        if not os.path.isdir(self.outputfolder):
            os.mkdir(self.outputfolder)
        plt.savefig( os.path.join(self.outputfolder,str(int(t)).zfill(maxzeros) + "_" +
            str(self.time_granularity)+".png"))
        plt.close(fig)

        #Plot derivatives
        if self.plot_derivatives:
            fig2 = plt.figure(figsize=(8,6))
            fig2.suptitle(str(int(t/60)) + " min")
            derivs = self.plate.get_cur_derivatives()
            #ax = Axes3D(fig)
            rad = np.linspace(0,self.plate_radius,self.radius_granularity)
            azm = np.linspace(0,2 * np.pi,self.angle_granularity)
            th,r = np.meshgrid(azm,rad)
            z = derivs[0]
            ax = plt.subplot(2,3,1,projection="polar")
            ax.set_title("dudt")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            z = derivs[1]
            ax = plt.subplot(2,3,2,projection="polar")
            ax.set_title("1/r*dudr")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            z = derivs[2]
            ax = plt.subplot(2,3,3,projection="polar")
            ax.set_title("du2dr2")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            z = derivs[3]
            ax = plt.subplot(2,3,4,projection="polar")
            ax.set_title("1/r^2*du2dtheta2")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            z = derivs[4]
            ax = plt.subplot(2,3,5,projection="polar")
            ax.set_title("light term")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            z = derivs[5]
            ax = plt.subplot(2,3,6,projection="polar")
            ax.set_title("decay term")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            plt.tight_layout()
            #plt.show()
            if not os.path.isdir(self.outputfolder):
                os.mkdir(self.outputfolder)
            plt.savefig( os.path.join(self.outputfolder,"derivs_" + str(int(t)).zfill(maxzeros) + "_" +
                str(self.time_granularity)+".png"))
            plt.close(fig2)


    def Step(self, t):
        cur_AHL_state = self.plate.get_cur_ahl_state()
        
        new_AHL_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        new_CI_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        new_Bgal_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        #f = open("debug.txt","w")
        for i in range(0,self.radius_granularity):
            #f.write("Radius: " + str(i) + "\n")
            for j in range(0,self.angle_granularity):
                new_AHL_state[i,j] = cur_AHL_state[i,j] + self.UpdateAHL_conc(cur_AHL_state,i,j) * self.dedimT(self.time_interval)
                new_CI_state[i,j] = self.k3 * Bacteria.f_light(self.light_mask[i,j])
                new_Bgal_state[i,j] = self.k4 * Bacteria.f_logic(new_AHL_state[i,j],new_CI_state[i,j])
                #f.write("Angle " + str(j) + ": " + str(new_Bgal_state[i,j]))
        #f.close()
        self.plate.AHL_history = new_AHL_state
        self.plate.CI_history = new_CI_state
        self.plate.Bgal_history = new_Bgal_state
        self.total_time += self.dedimT(self.time_interval)

    def UpdateAHL_conc(self,cur_state,i,j):
        dudt = 0
        dudr = 0
        du2dr2 = 0
        du2dtheta2 = 0
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
            dudr = (cur_state[forwardradius,j] - cur_state[backwardradius,int(j+self.angle_granularity/2.0) % self.angle_granularity]) / (2.0 * self.dedimR(self.radius_interval)) 
            du2dr2 = (cur_state[forwardradius,j] - 2.0 * cur_state[i,j] + cur_state[backwardradius,int(j+self.angle_granularity/2.0) % self.angle_granularity]) / (self.dedimR(self.radius_interval) ** 2.0)
            #new_state += (cur_state[forwardradius,j] - cur_state[backwardradius,j]) / (2 * self.radius_interval) / self.dedimR(self.radius_h[i])
            #new_state += (cur_state[forwardradius,j] - 2 * cur_state[i,j] + cur_state[backwardradius,j]) / (self.radius_interval ** 2)
        elif i == (self.radius_granularity - 1):
            dudr = (cur_state[i,j] - cur_state[i-1,j]) / self.dedimR(self.radius_interval)
            du2dr2 = (cur_state[i,j] - 2.0 * cur_state[i-1,j] + cur_state[i-2,j]) / (self.dedimR(self.radius_interval) ** 2.0)
            #new_state += (3 * cur_state[i,j] - 4 * cur_state[i-1,j] + cur_state[i-2,j]) / (2 * self.radius_interval) / self.dedimR(self.radius_h[i])
            #new_state += (2 * cur_state[i,j] - 5 * cur_state[i-1,j] + 4 * cur_state[i-2,j] - cur_state[i-3,j]) / (self.radius_interval ** 2)
        else:
            dudr = (cur_state[forwardradius,j] - cur_state[backwardradius,j]) / (2.0 * self.dedimR(self.radius_interval))
            du2dr2 = (cur_state[forwardradius,j] - 2.0 * cur_state[i,j] + cur_state[backwardradius,j]) / (self.dedimR(self.radius_interval) ** 2.0)

        du2dtheta2 = (cur_state[i,forwardangle] - 2.0 * cur_state[i,j] + cur_state[i,backwardangle]) / (self.angle_interval ** 2.0) / (self.dedimR(self.radius_h[i]) ** 2.0)
        #if (cur_state[i,forwardangle] - 2 * cur_state[i,j] + cur_state[i,backwardangle]) != 0:
        #    print("radius:" + str(i) + ", angle:" + str(j) + ", curstate:" + str(cur_state[i,j]) + ", val:" + str((cur_state[i,forwardangle] - 2 * cur_state[i,j] + cur_state[i,backwardangle])))
        #print(np.random.normal(loc=0, scale =0.05*self.k1))
        #print(np.random.normal(loc=0, scale=0.05*self.k2))
        #print((self.k1 + np.random.normal(loc=0, scale =0.05*self.k1))/self.k1)
        #print((self.k2 + np.random.normal(loc=0, scale=0.05*self.k2))/self.k2)
        dudt += 1.0/self.dedimR(self.radius_h[i])*dudr + du2dr2 + 1.0/(self.dedimR(self.radius_h[i]) ** 2.0)*du2dtheta2 + (self.k1) * Bacteria.f_light(self.light_mask[i,j]) - (self.k2) * cur_state[i,j]

        self.plate.dudt[i,j] = dudt;
        self.plate.dudr[i,j] = dudr * 1.0/self.dedimR(self.radius_h[i]);
        self.plate.du2dr2[i,j] = du2dr2;
        self.plate.du2dtheta2[i,j] = du2dtheta2 *1.0/(self.dedimR(self.radius_h[i]) ** 2.0);
        self.plate.light_term[i,j] = (self.k1) * Bacteria.f_light(self.light_mask[i,j])
        self.plate.decay_term[i,j] = - (self.k2) * cur_state[i,j]
        return dudt

sim = Simulation(sys.argv[1:])
sim.run()