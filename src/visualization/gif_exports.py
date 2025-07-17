plt.figure(figsize=(8, 4))
plt.plot(x, u_history[-1], label="Final u(x, t)", color='red')
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("Final Heat Distribution")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figures/final_snapshot.png", dpi=300)
plt.show()

def compute_l2_error(u_num, u_ref, dx):
    return np.sqrt(simpson((u_num - u_ref)**2, dx=dx))

u_ref = u0.copy()
l2_error = compute_l2_error(u_history[-1], u_ref, dx)