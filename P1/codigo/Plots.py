import matplotlib.pyplot as plt

def plot_recompensas(recompensas, pasos_por_episodio):
    plt.figure(figsize=(12, 5))
    plt.plot(recompensas, label='Reward per Step')

    for step in range(pasos_por_episodio, len(recompensas), pasos_por_episodio):
        plt.axvline(x=step, color='gray', linestyle='--', linewidth=0.5)

    plt.xlabel("Paso")
    plt.ylabel("Recompensa")
    #plt.title()
    plt.grid(True)
    #plt.legend()
    plt.show()

