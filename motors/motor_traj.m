clear
theta = [0 0.1 0.2 0.3;
        0 0.1 0.2 0.3;
        0 -0.1 -0.2 -0.3];

shoulder1 = [180 160 90 180	200	180];
shoulder2 = [180 210 210 140 160 180];
shoulder3 = repmat(270,1,length(shoulder2));
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = repmat(360,1,length(shoulder2));

shoulder1 = [180 240 110 180];
shoulder2 = [180 270 150 180];
shoulder3 = repmat(270,1,length(shoulder2));
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 180];

timePoints = [0 1 3 3.5];


wayPoints = [shoulder4; shoulder5; shoulder6]*pi/180;
% wayPoints(wayPoints < 0) = wayPoints(wayPoints < 0) + 360;
tvec = 0:0.01:timePoints(end);
N = length(tvec);
[qd,dqd,ddqd,pp] = cubicpolytraj(wayPoints,timePoints,tvec);

save('qd.mat','qd');
save('dqd.mat','dqd');
save('ddqd.mat','ddqd');
save('tvec.mat','tvec');