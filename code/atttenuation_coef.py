#%%
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
x_val_oxygen = 1000 * oxygen_coef[:,0]
x_val_carbon = 1000 * carbon_coef[:,0]
x_val_hydrogen = 1000 * hydrogen_coef[:,0]

y_val_oxygen = oxygen_coef[:,1]
y_val_carbon = carbon_coef[:,1]
y_val_hydrogen = hydrogen_coef[:,1]

x_val_wood = wood_composite[0]*x_val_carbon+wood_composite[1]*x_val_hydrogen+wood_composite[2]*x_val_oxygen
y_val_wood = wood_composite[0]*y_val_carbon+wood_composite[1]*y_val_hydrogen+wood_composite[2]*y_val_oxygen


xinterp = np.linspace(min(x_val_oxygen), max(x_val_oxygen), 1000)
# print(f"xinterp: {xinterp}")
#yinterp_wood = wood_composite[0]*np.interp(xinterp, x_val_carbon, y_val_carbon)+wood_composite[1]*np.interp(xinterp, x_val_hydrogen, y_val_hydrogen)+wood_composite[2]*np.interp(xinterp, x_val_oxygen, y_val_oxygen)



# Change from MeV to KeV
x_val_iron = 1000 * iron_coef[:,0]
x_val_bismuth = 1000 * bismuth_coef[:,0]
x_val_lead = 1000 * lead_coef[:, 0]

# print(f"bismuth x: {x_val_bismuth}")
# print(f"log bismuth x: {np.log(x_val_bismuth)}")

y_val_iron = iron_coef[:,1]
y_val_bismuth = bismuth_coef[:,1]
y_val_lead = lead_coef[:, 1]

#Interpolate data

yinterp_bismuth = np.interp(xinterp, x_val_bismuth, y_val_bismuth)

yinterp_iron = np.interp(xinterp, x_val_iron, y_val_iron)

yinterp_lead = np.interp(xinterp, x_val_lead, y_val_lead)

yinterp_wood = np.interp(xinterp, x_val_wood, y_val_wood)

diff_coef = np.abs(yinterp_bismuth-yinterp_iron)



x_combined = sorted(list(x_val_iron) + list(x_val_bismuth))
new_y_bismuth = np.interp(x_combined, x_val_bismuth, y_val_bismuth)
new_y_iron = np.interp(x_combined, x_val_iron, y_val_iron)

diff_yinterp = np.abs(new_y_bismuth - new_y_iron)
plt.figure("diff")
plt.plot(
    np.log([x for x in x_combined if x >= 10 and x <= 200]), 
    [diff for x,diff in zip(x_combined, diff_yinterp) if x >= 10 and x <= 200], '-x')
plt.xlabel("Log X-Ray [keV]")
plt.ylabel("Attenuation coef [cm^2 / g]")
plt.legend(["Absolute difference between bismuth and iron"])

# plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)

combined = [(diff, x, np.log(x)) for diff,x in zip(diff_yinterp, x_combined) if x >= 10 and x <= 200]
print(max(combined, key=lambda v: v[0]))

# plt.figure("log x-axis")
# #plt.plot(np.log(x_val_bismuth), np.log(y_val_bismuth), 'o')
# #plt.plot(np.log(x_val_iron), (y_val_iron), 'o')
# plt.plot(np.log(xinterp),yinterp_bismuth, '-x')
# plt.plot(np.log(xinterp),yinterp_iron, '-x')
# plt.plot(np.log(xinterp),yinterp_wood,'-x')
# plt.plot(np.log(xinterp),yinterp_lead, '-x')

# plt.xlabel("Log X-Ray [KeV]")
# plt.ylabel("Attenuation coefficients [cm^2 / g]")

# plt.legend(["Bismuth", "Iron", "Wood", "Lead"])

# plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)

# iron_bismuth_diff = np.abs(y_val_bismuth - y_val_iron)

diff_f = lambda x_list, diff_list: [diff for x, diff in zip(x_list, diff_list) if x >= 10 and x <= 200]
x_f = lambda x_list: [x for x in x_list if x >= 10 and x <= 200]

plt.figure("raw data, log x")
plt.plot(np.log(x_f(x_val_iron)), diff_f(x_val_iron, y_val_iron), '-x')
plt.plot(np.log(x_f(x_val_bismuth)), diff_f(x_val_bismuth, y_val_bismuth), '-x')
plt.plot(np.log(x_f(x_val_wood)), diff_f(x_val_wood, y_val_wood), '-x')
plt.xlabel("Log X-Ray [keV]")
plt.ylabel("Attenuation coef [cm^2 / g]")

#plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)
plt.legend(["Iron", "Bismuth", "Wood"])

plt.show()

# %%
