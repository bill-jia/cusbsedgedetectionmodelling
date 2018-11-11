#!/usr/bin/python
import sys, os, argparse, time
from datetime import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate
import skimage as ski
import skimage.io
import skimage.transform
import skimage.filters

np.seterr(all='raise')

def floatrange(start,end,inter):
    output = [start]
    while output[len(output)-1] < end:
        output.append(output[len(output)-1] + inter)
    return output

class Bacteria:
    '''
        This class is responsible for modelling the dynamics of the gene expression circuit.
        It holds all of the constants and the transfer functions that decide an output based
        on the current state of the cell.
    '''
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
    """
    This class stores spatial information about the media, i.e. the chemical concentrations
    at every point and their derivatives across space and time, required for update of the
    differential equations.
    """
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
    """
    This class runs the simulations
    """
    def __init__(self,argv):
        #Initialize adjustable parameters according to command line input
        parser = argparse.ArgumentParser(description="Edge detection modelling")
        parser.add_argument('--opath', '-o', nargs='?', default = datetime.now().strftime("%y%m%d-%H-%M-%S"))
        parser.add_argument('--plot', '-p', action='store_true')
        parser.add_argument('--ginterval', '-g', nargs='?', type=int, default = 10)
        parser.add_argument('--dplot', '-d', action='store_true')
        parser.add_argument('--end_time', '-t', nargs='?', type=float, default=24, help='End time in hours')
        parser.add_argument('--tfs', '-f', action='store_true')
        parser.add_argument('--maskpath', '-m', nargs='?', default=None)
        parser.add_argument('--maxlight', '-l', nargs='?', type=float, default = 0.15)
        parser.add_argument('--radius_granularity', '-r', nargs='?', type=int, default = 100)
        parser.add_argument('--angle_granularity', '-a', nargs='?', type=int, default = 100)
        args = parser.parse_args()
        self.outputfolder = args.opath
        self.plots = args.plot
        self.graph_interval = args.ginterval
        self.max_time = args.end_time*60*60.0
        self.plot_derivatives = args.dplot
        self.plot_transfer_functions = args.tfs
        self.mask_path = args.maskpath
        self.max_light = args.maxlight
        self.radius_granularity = args.radius_granularity
        self.angle_granularity = args.angle_granularity

        #Granularity constants
        self.time_granularity = self.radius_granularity*2 # Time step twice as fine as space step
 
        
        #Setup initial conditions
        self.AHL_Diffusion_Coef = 1.67 * (10 ** (-7)) # cm^2/s
        self.plate_radius = 4.25 # cm
        self.k1 = 0.03/289 / 3600 / (self.AHL_Diffusion_Coef/self.plate_radius**2) # nM/hr converted to nM/dimensionless time
        self.k2 = 0.012 / 3600 / (self.AHL_Diffusion_Coef/self.plate_radius**2) # hr^-1 converted to 1/dimensionless time
        self.k3 = 0.8 # nM/Miller
        self.k4 = 289.0 # Miller units
        print("k1= ", self.k1)
        print("k2= ", self.k2)
        self.time_interval = self.max_time / self.time_granularity
        self.radius_interval = self.plate_radius / self.radius_granularity
        self.angle_interval = 2.0 * np.pi / self.angle_granularity

        # Generate sampling points
        # Sampling points start at 1/2 radius_interval
        self.radius_h = floatrange(self.radius_interval/2, self.plate_radius - self.radius_interval/2, self.radius_interval)
        self.angle_h = floatrange(0, 2 * np.pi - self.angle_interval, self.angle_interval)
        self.time_h = floatrange(self.time_interval, self.max_time, self.time_interval)
        #self.time_h = floatrange(self.time_interval, 24 * 3600, self.time_interval)

        #Initialize plate
        self.plate = Media(self.radius_granularity, self.angle_granularity)
        #Set light input
        self.light_mask = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        if self.mask_path == None:
            # If default mask, generate a spot of light half the radius of the plate
            for i in range(0,int(self.radius_granularity/2)):
                for j in range(0,self.angle_granularity):
                    self.light_mask[i,j] = self.max_light
        else:
            #If a greyscale image mask is specified, size image according to granularities and normalize to values between 0 and max light
            
            # Read image
            img = ski.io.imread(self.mask_path)
            
            # Scale image to resolution of plate radius
            (y, x) = img.shape
            scaling_factor = (((x/2)**2 + (y/2)**2)**(0.5))/(self.radius_granularity*0.95)
            img = ski.filters.gaussian(img, (1-1/scaling_factor)/2, preserve_range=True)

            # Sample image to populate light mask
            for i in range(0,int(self.radius_granularity)):
                for j in range(0,self.angle_granularity):
                    self.light_mask[i,j] = (self.interp(img, i, j, scaling_factor)/255.0)*self.max_light
            
            fig3 = plt.figure()
            rad = np.linspace(0,self.plate_radius,self.radius_granularity)
            azm = np.linspace(0,2 * np.pi,self.angle_granularity)
            th,r = np.meshgrid(azm,rad)
            z = self.light_mask
            ax = plt.subplot(1,1,1,projection="polar")
            ax.set_title("Mask")
            colors = plt.pcolormesh(th,r,z)
            plt.colorbar(colors)
            plt.grid()
            plt.show()

        self.total_time = 0

    def interp(self, img, i, j, scaling_factor):
        (y_max, x_max) = img.shape
        x_samp = (i * np.cos(j*2*np.pi/self.angle_granularity))*scaling_factor + x_max/2.0
        y_samp = y_max/2.0 - (i * np.sin(j*2*np.pi/self.angle_granularity))*scaling_factor
        #print(x_samp, y_samp)
        if (x_samp < 0 or x_samp > x_max - 1) or (y_samp < 0 or y_samp > y_max - 1):
            return 0
        else:
            if (x_samp == np.floor(x_samp)) or (y_samp == np.floor(y_samp)):
                i_samp = img[int(y_samp), int(x_samp)]
            else:
                x1 = int(np.floor(x_samp))
                x2 = int(np.ceil(x_samp))
                y1 = int(np.floor(y_samp))
                y2 = int(np.ceil(y_samp))
                a = (y1, x1)
                b = (y1, x2)
                c = (y2, x1)
                d = (y2, x2)
                #print (a, b, c, d, (y_samp, x_samp))
                i_samp = ((img[a]*(x2-x_samp) + img[b]*(x_samp-x1))*(y2-y_samp) + (img[c]*(x2-x_samp) + img[d]*(x_samp-x1))*(y_samp-y1))/((x2-x1)*(y2-y1))
            if i_samp < 0:
                print(i_samp)
                print(a,b,c,d)
            return i_samp




    def dedimR(self, r):
        """
        Dedimensionalise radius
        """
        return r / self.plate_radius

    def dedimT(self, t):
        """
        Dedimensionalise time
        """
        return t * self.AHL_Diffusion_Coef / (self.plate_radius ** 2)

    def run(self):
        count = 0
        if self.plot_transfer_functions:
            self.plot_tfs()
            

        print("Simulating to: " + str(max(self.time_h)/60) + " minutes")
        if self.plots:
            print("Plotting every: " + str(self.graph_interval) + " timepoints")
        else:
            print("No plots being made")
        print("Each time step: " + str(self.time_interval/60) + " minutes")
        print("Dimensionless radius interval: ", self.dedimR(self.radius_interval))
        print("Dimensionless time interval: ", self.dedimT(self.time_interval))
        print("Theta interval: ", self.angle_interval)
        
        for t in self.time_h:
            #print(t)
            start = time.time()
            self.Step(t)
            count += 1
            if count % self.graph_interval == 0 and self.plots :
                self.make_plots(t);
            end = time.time()
            #print("Last step took " + str(end-start) + " seconds", end="\r")
        print(self.total_time)
        cur_state = self.plate.get_cur_state()
        print("Maximum Bgal concentration (Miller): ", cur_state[1].max())
        print("Maximum AHL concentration (nM): ", cur_state[0].max())

    def Step(self, t):
        """
        This function steps through time and updates the chemical concentrations based on the equations.
        """
        cur_AHL_state = self.plate.get_cur_ahl_state()
        
        new_AHL_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        new_CI_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        new_Bgal_state = np.zeros(shape=(self.radius_granularity,self.angle_granularity))
        for i in range(0,self.radius_granularity):
            for j in range(0,self.angle_granularity):
                new_AHL_state[i,j] = cur_AHL_state[i,j] + self.UpdateAHL_conc(cur_AHL_state,i,j) * self.dedimT(self.time_interval)
                new_CI_state[i,j] = self.k3 * Bacteria.f_light(self.light_mask[i,j])
                new_Bgal_state[i,j] = self.k4 * Bacteria.f_logic(new_AHL_state[i,j],new_CI_state[i,j])
        self.plate.AHL_history = new_AHL_state
        self.plate.CI_history = new_CI_state
        self.plate.Bgal_history = new_Bgal_state
        self.total_time += self.dedimT(self.time_interval)

    def UpdateAHL_conc(self,cur_state,i,j):
        """
        Calculates the rate change over time of AHL according to the change over space at each time
        point (reaction-diffusion equation)
        """
        dudt = 0
        dudr = 0
        du2dr2 = 0
        du2dtheta2 = 0
        # Get the radius and angle (accounting for the fact that the plate is a circle)
        forwardradius, backwardradius, forwardangle, backwardangle = self.get_radius_angles(i, j)

        # Solve the spatial derivatives
        if i == 0:
            dudr = 0 # Flux across the pole cancels out at first order
            du2dr2 = np.around((2.0*cur_state[forwardradius,j] - 2.0 * cur_state[i,j]), decimals=5) / (self.dedimR(self.radius_interval) ** 2.0)
        elif i == (self.radius_granularity - 1):
            dudr = (cur_state[i,j] - cur_state[backwardradius,j]) / (2.0 * self.dedimR(self.radius_interval))
            du2dr2 = np.around((cur_state[backwardradius,j]-cur_state[i,j]), decimals=5) / (self.dedimR(self.radius_interval) ** 2.0)
        else:
            dudr = (cur_state[forwardradius,j] - cur_state[backwardradius,j]) / (2.0 * self.dedimR(self.radius_interval))
            du2dr2 = np.around((cur_state[forwardradius,j] - 2.0 * cur_state[i,j] + cur_state[backwardradius,j]), decimals=5) / (self.dedimR(self.radius_interval) ** 2.0)

        du2dtheta2 = np.around((cur_state[i,forwardangle] - 2.0 * cur_state[i,j] + cur_state[i,backwardangle]), decimals=3) / (self.angle_interval ** 2.0)
        
        du2dtheta2 = np.around(du2dtheta2, decimals=3)

        # Update the time derivative according to the reaction-diffusion equation
        dudt += 1.0/self.dedimR(self.radius_h[i])*dudr + du2dr2 + 1.0/(self.dedimR(self.radius_h[i]) ** 2.0)*du2dtheta2 + (self.k1) * Bacteria.f_light(self.light_mask[i,j]) - (self.k2) * cur_state[i,j]

        self.plate.dudt[i,j] = dudt
        self.plate.dudr[i,j] = dudr * 1.0/self.dedimR(self.radius_h[i])
        self.plate.du2dr2[i,j] = du2dr2
        self.plate.du2dtheta2[i,j] = du2dtheta2 *1.0/(self.dedimR(self.radius_h[i]) ** 2.0)
        self.plate.light_term[i,j] = (self.k1) * Bacteria.f_light(self.light_mask[i,j])
        self.plate.decay_term[i,j] = - (self.k2) * cur_state[i,j]
        return dudt

    def get_radius_angles(self, i, j):
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
        return forwardradius, backwardradius, forwardangle, backwardangle

    def plot_tfs(self):
        # Plot transfer functions if flag is true
        light = floatrange(0, 0.1, 0.0001)
        f_light = [Bacteria.f_light(l) for l in light]
        tf_1 = plt.figure()
        ax1 = tf_1.add_subplot(1,1,1)
        ax1.set_title("f_light")
        ax1.set_xlabel("Light Intensity(W/m2)")
        ax1.set_ylabel("Miller Units")
        ax1.plot(light, f_light)
        
        ahl = np.geomspace(0.1, 250, 200)
        ci = np.linspace(120, 350, 100)
        ahlv, civ = np.meshgrid(ahl, ci)
        # Note that CI concentration for y-axis is defined in Millers (i.e. output of f_light)
        f_logic = Bacteria.f_logic(ahlv, civ*self.k3)
        tf_2 = plt.figure()
        ax2 = tf_2.add_subplot(1,1,1)
        ax2.set_title("f_logic")
        ax2.set_xscale("log")
        ax2.set_xlabel("AHL Concentration (nM)")
        ax2.set_ylabel("CI Concentration (Miller Units)")
        plt.contourf(ahlv, civ, f_logic, 30, cmap="jet", vmin=0, vmax=0.35)
        plt.colorbar()
        plt.show()

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

sim = Simulation(sys.argv[1:])
sim.run()