# 3d2d_visual_odometry

Implementação de um algoritmo de Odometria Visual para o Kitti Dataset. Funcionamento:

- Leitura das imagens
- Detecção de características (FAST)
- Matching das das características no par de imagens (BRIEF)
- Triangulação
- Tracking das características na próxima imagem
- Cálculo da transformação de corpo rígido (SolvePnP)
- Concatenação da tragetória

Resultados para a rota 00 do Kitti Dataset:
![Kitti-00](https://github.com/pmec-home/3d2d_visual_odometry/blob/main/00.png)
