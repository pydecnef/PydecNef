import pandas as pd
import numpy as np
def generating_noisy_signal(data, target_snr_db, mean_noise):
    signal = pd.DataFrame(data)
    signal_power = signal ** 2
    signal_avg_power = signal_power.mean(axis=1)
    sig_avg_db = 10 * np.log10(signal_avg_power)
    noise_avg_db = sig_avg_db - float(target_snr_db)
    noise_avg_power = 10 ** (noise_avg_db / 10)
    noise_power = np.random.normal(mean_noise, np.sqrt(noise_avg_power), len(signal_power))
    noisy_signal = signal.add(noise_power,axis=0)
    return noisy_signal