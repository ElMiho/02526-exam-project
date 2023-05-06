#%%
import numpy as np
import matplotlib.pyplot as plt

iron_coef = np.loadtxt('iron_data.txt')
bismuth_coef = np.loadtxt('bismuth_data.txt')
oxygen_coef = np.loadtxt('oxygen_data.txt')
carbon_coef = np.loadtxt('carbon_data.txt')
nitrogen_coef = np.loadtxt('nitrogen_data.txt')
hydrogen_coef = np.loadtxt('hydrogen_data.txt')
lead_coef = np.loadtxt('lead_data.txt')

#Carbon, hydrogen, oxygen
wood_composite = [0.5, 0.0615, 0.43, 0.09]
x_val_oxygen = 1000 * oxygen_coef[:,0]
x_val_carbon = 1000 * carbon_coef[:,0]
x_val_hydrogen = 1000 * hydrogen_coef[:,0]
x_val_nitrogen = 1000 * nitrogen_coef[:,0]

y_val_oxygen = oxygen_coef[:,1]
y_val_carbon = carbon_coef[:,1]
y_val_hydrogen = hydrogen_coef[:,1]
y_val_nitrogen = nitrogen_coef[:,1]


# The energies
x_val_wood = wood_composite[0]*x_val_carbon+wood_composite[1]*x_val_hydrogen+wood_composite[2]*x_val_oxygen+wood_composite[3]*x_val_nitrogen
# mu / rho
y_val_wood = wood_composite[0]*y_val_carbon+wood_composite[1]*y_val_hydrogen+wood_composite[2]*y_val_oxygen+wood_composite[3]*y_val_nitrogen

# plt.plot(np.log(x_val_wood), np.log(y_val_wood))
# plt.xlabel("Log Energy [keV]")
# plt.ylabel(r"$\mu / \rho$")
# plt.show()

# Plots energy used vs proportion of beam intensity leaving log to be recepted compared to incomming
I0 = 1
density_wood = 0.71 # g / cm^3
thickness = 50 # cm
I = lambda energy: I0 * np.exp(-y_val_wood[list(x_val_wood).index(energy)] * (thickness * density_wood))

my_if = lambda x: x >= 5 and x <= 220
plt.plot([x for x in x_val_wood if my_if(x)], [I(x) * 100 for x in x_val_wood if my_if(x)])
plt.plot(200, 0.6, "*")
plt.xlabel("Energy [keV]")
plt.ylabel("Output intensity [%]")
plt.show()

# xinterp = np.linspace(min(x_val_oxygen), max(x_val_oxygen), 1000)
# # print(f"xinterp: {xinterp}")
# #yinterp_wood = wood_composite[0]*np.interp(xinterp, x_val_carbon, y_val_carbon)+wood_composite[1]*np.interp(xinterp, x_val_hydrogen, y_val_hydrogen)+wood_composite[2]*np.interp(xinterp, x_val_oxygen, y_val_oxygen)


# #Density, wood, iron, lead, bismuth [g/cm^3]
# density = [0.82, 7.874, 11.34, 9.78]

# # Change from MeV to KeV
# x_val_iron = 1000 * iron_coef[:,0]
# x_val_bismuth = 1000 * bismuth_coef[:,0] 
# x_val_lead = 1000 * lead_coef[:, 0]

# # print(f"bismuth x: {x_val_bismuth}")
# # print(f"log bismuth x: {np.log(x_val_bismuth)}")

# y_val_iron = iron_coef[:,1] * density[1]
# y_val_bismuth = bismuth_coef[:,1] * density[3]
# y_val_lead = lead_coef[:, 1] * density[2]

# y_val_wood = density[0] * y_val_wood

# #Interpolate data

# yinterp_bismuth = np.interp(xinterp, x_val_bismuth, y_val_bismuth)

# yinterp_iron = np.interp(xinterp, x_val_iron, y_val_iron)

# yinterp_lead = np.interp(xinterp, x_val_lead, y_val_lead)

# yinterp_wood = np.interp(xinterp, x_val_wood, y_val_wood)

# diff_coef = np.abs(yinterp_bismuth-yinterp_iron)



# x_combined = sorted(list(x_val_iron) + list(x_val_bismuth))
# new_y_bismuth = np.interp(x_combined, x_val_bismuth, y_val_bismuth)
# new_y_iron = np.interp(x_combined, x_val_iron, y_val_iron)
# new_y_wood = np.interp(x_combined, x_val_wood, y_val_wood)

# diff_yinterp = np.abs(new_y_bismuth - new_y_iron)
# plt.figure("diff")
# plt.plot(
#     np.log([x for x in x_combined if x >= 10 and x <= 200]), 
#     [diff for x,diff in zip(x_combined, diff_yinterp) if x >= 10 and x <= 200], '-x', linewidth=3)
# plt.xlabel("Log X-Ray [keV]")
# plt.ylabel("Attenuation coef [cm^(-1)]")
# plt.legend(["Absolute difference between bismuth and iron"])

# # plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)
# plt.savefig(".././images/diff-attenuation-coef-bismuth-iron-zoom.png")
# plt.close()

# combined = [(diff, x, np.log(x)) for diff,x in zip(diff_yinterp, x_combined) if x >= 10 and x <= 200]
# print(max(combined, key=lambda v: v[0]))
# for x, bismuth_atten_coeff, iron_atten_coeff, wood_atten_coeff in zip(x_combined, new_y_bismuth, new_y_iron, new_y_wood):
#     if x == max(combined, key=lambda v: v[0])[1]:
#         print(f"Bismuth coeff: {bismuth_atten_coeff} | Iron coeff: {iron_atten_coeff} | Wood coeff: {wood_atten_coeff}")
#         break

# # plt.figure("log x-axis")
# # #plt.plot(np.log(x_val_bismuth), np.log(y_val_bismuth), 'o')
# # #plt.plot(np.log(x_val_iron), (y_val_iron), 'o')
# # plt.plot(np.log(xinterp),yinterp_bismuth, '-x')
# # plt.plot(np.log(xinterp),yinterp_iron, '-x')
# # plt.plot(np.log(xinterp),yinterp_wood,'-x')
# # plt.plot(np.log(xinterp),yinterp_lead, '-x')

# # plt.xlabel("Log X-Ray [KeV]")
# # plt.ylabel("Attenuation coefficients [cm^2 / g]")

# # plt.legend(["Bismuth", "Iron", "Wood", "Lead"])

# # plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)

# # iron_bismuth_diff = np.abs(y_val_bismuth - y_val_iron)

# diff_f = lambda x_list, diff_list: [diff for x, diff in zip(x_list, diff_list) if x >= 10 and x <= 200]
# x_f = lambda x_list: [x for x in x_list if x >= 10 and x <= 200]

# plt.figure("raw data, log x")
# plt.plot(np.log(x_f(x_val_iron)), diff_f(x_val_iron, y_val_iron), '-x', linewidth=3)
# plt.plot(np.log(x_f(x_val_bismuth)), diff_f(x_val_bismuth, y_val_bismuth), '-x', linewidth=3)
# plt.plot(np.log(x_f(x_val_wood)), diff_f(x_val_wood, y_val_wood), '-x', linewidth=3)
# plt.xlabel("Log X-Ray [keV]")
# plt.ylabel("Attenuation coef [cm^(-1)]")

# #plt.fill_betweenx([min(y_val_iron),max(y_val_iron)],np.log(10),np.log(200),alpha=0.5)
# plt.legend(["Iron", "Bismuth", "Wood"])

# plt.savefig(".././images/attenuation_coef_zoom.png")
# plt.close()
# # %%

# %%
