clear
theta = [0 0.1 0.2 0.3;
        0 0.1 0.2 0.3;
        0 -0.1 -0.2 -0.3];

shoulder1 = [180 160 90 180	200	180];
shoulder2 = [180 210 210 140 160 180];
shoulder3 = repmat(270,1,length(shoulder2));
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = repmat(0,1,length(shoulder2));

timePoints = [0 0.5 1 2.5 3 3.5];


wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6];
% wayPoints(wayPoints < 0) = wayPoints(wayPoints < 0) + 360;
tSamples = 0:0.01:timePoints(end);
N = length(tSamples);
[qd,dqd,ddqd,pp] = cubicpolytraj(wayPoints,timePoints,tSamples);

save('q.mat','qd');
save('tvec.mat','tSamples');