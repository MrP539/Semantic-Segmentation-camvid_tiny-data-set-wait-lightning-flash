import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# อ่านข้อมูลจาก CSV
csv_path = 'logs/lightning_logs/version_0/metrics.csv'
metrics = pd.read_csv(f'{csv_path}')
del metrics["step"]
metrics.set_index("epoch", inplace=True)

# สร้างกราฟเส้นโดยไม่มีแถบความเชื่อมั่น
g = sns.relplot(data=metrics, kind="line", errorbar=None)
g.fig.set_size_inches(12, 6)  # ปรับขนาดกราฟ
plt.grid()  # เพิ่มกริด

# บันทึกเป็นไฟล์ PNG
plt.savefig("metrics_plot.png")