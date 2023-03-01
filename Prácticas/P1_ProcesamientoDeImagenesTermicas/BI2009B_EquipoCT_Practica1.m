%% BI2009B Procesamiento de Imágenes Médicas para el Diagnóstico
%% Práctica 1. Procesamiento de Imágenes Termales
%% Equipo CT

clear
clc
close all

% Lectura de la Imagen a Procesar
f=imread("DiegoTermal_2.jpg");
info = imfinfo("DiegoTermal_2.jpg");

t = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');

% Imagen Térmica con Escala de Grises
f = double(f(:,:,1));
f = f / max(max(f));
fx = nexttile;
imshow(f,[]);
gmap = colormap(fx,gray);
k = 0.56;
addMM=@(x) sprintf('%.0fmm',x * k);
axis on
xticklabels(cellfun(addMM,num2cell(xticks'),'UniformOutput',false));
yticklabels(cellfun(addMM,num2cell(yticks'),'UniformOutput',false));
colorbar;
title("Imagen Térmica con Escala de Grises","FontSize",14)

% Histograma de Imagen Térmica con Escala de Grises
fx2 = nexttile;
[histdigital,binloc] = imhist(f);
area(binloc,histdigital);
title("Histograma  - Imagen Térmica con Escala de Grises","FontSize",14)
xlabel("Niveles de Gris (Bins)")
ylabel("Conteo")
grid on

% Conversión a Grados Centígrados
mintemp = 29;
maxtemp = 39;
temcent = (f-min(min(f)))/(max(max(f))-min(min(f)));
temcent = temcent*(maxtemp-mintemp)+mintemp;

% Imagen Térmica con Mapeo de Colores
fx3 = nexttile(t,3);
imshow(temcent,[mintemp,maxtemp]);
axis on
xticklabels(cellfun(addMM,num2cell(xticks'),'UniformOutput',false));
yticklabels(cellfun(addMM,num2cell(yticks'),'UniformOutput',false));
cmap = colormap(fx3,hot);
colorbar;
title("Imagen Térmica con Mapeo de Colores","FontSize",14)

% Histograma de Imagen Térmica con Mapeo de Colores
fx3 = nexttile(t,4);
[histcent,binloc] = imhist(temcent,cmap);
area(binloc,histcent);
title("Histograma  - Imagen Térmica con Mapeo de Colores y Conversión a Grados Centígrados","FontSize",14)
xlabel("Niveles de Color (Bins)")
ylabel("Conteo")
xlim([29 39])
grid on

% Cálculo Distancia Entre Pupilas
ojo1 = 222;
ojo2 = 349;
distOjos = abs(ojo1 - ojo2) * k;