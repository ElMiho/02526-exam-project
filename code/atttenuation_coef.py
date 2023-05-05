import numpy as np
import matplotlib.pyplot as plt

iron_coef = np.loadtxt('iron_data.txt')
bismuth_coef = np.loadtxt('bismuth_data.txt')
oxygen_coef = np.loadtxt('oxygen_data.txt')
carbon_coef = np.loadtxt('carbon_data.txt')
hydrogen_coef = np.loadtxt('hydrogen_data.txt')
lead_coef = np.loadtxt('lead_data.txt')

#Carbon, hydrogen, oxygen
wood_composite = [0.5, 0.06, 0.44]
x_val_oxygen = 1000*oxygen_coef[:,0]
x_val_carbon = 1000*carbon_coef[:,0]
x_val_hydrogen = 1000*hydrogen_coef[:,0]

y_val_oxygen = oxygen_coef[:,1]
y_val_carbon = carbon_coef[:,1]
y_val_hydrogen = hydrogen_coef[:,1]

xinterp = np.linspace(min(x_val_oxygen), max(x_val_oxygen), 1000)
yinterp_wood = wood_composite[0]*np.interp(xinterp, x_val_carbon, y_val_carbon)+wood_composite[1]*np.interp(xinterp, x_val_hydrogen, y_val_hydrogen)+wood_composite[2]*np.interp(xinterp, x_val_oxygen, y_val_oxygen)


# Change from MeV to KeV
x_val_iron = 1000*iron_coef[:,0]
x_val_bismuth = 1000*bismuth_coef[:,0]
x_val_lead = 1000 * lead_coef[:, 0]

y_val_iron = iron_coef[:,1]
y_val_bismuth = bismuth_coef[:,1]
y_val_lead = lead_coef[:, 1]

#Interpolate data

yinterp_bismuth = np.interp(xinterp, x_val_bismuth, y_val_bismuth)

yinterp_iron = np.interp(xinterp, x_val_iron, y_val_iron)

yinterp_lead = np.interp(xinterp, x_val_lead, y_val_lead)

diff_coef = np.abs(yinterp_bismuth-yinterp_iron)


plt.figure(1)
#plt.plot(np.log(x_val_bismuth), np.log(y_val_bismuth), 'o')
#plt.plot(np.log(x_val_iron), (y_val_iron), 'o')
plt.plot(np.log(xinterp),yinterp_bismuth, '-x')
plt.plot(np.log(xinterp),yinterp_iron, '-x')
plt.plot(np.log(xinterp),yinterp_wood,'-x')
plt.plot(np.log(xinterp),yinterp_lead, '-x')

plt.xlabel("Log X-Ray [KeV]")
plt.ylabel("Attenuation coefficients [cm^2 / g]")

plt.legend(["Bismuth", "Iron", "Wood", "Lead"])

plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)

plt.show()
