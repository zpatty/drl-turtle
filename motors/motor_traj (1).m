clear
theta = [0 0.1 0.2 0.3;
        0 0.1 0.2 0.3;
        0 -0.1 -0.2 -0.3];

shoulder1 = [180 160 90 180	200	180];
shoulder2 = [180 210 210 140 160 180];
shoulder3 = repmat(180,1,length(shoulder2));
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = repmat(180,1,length(shoulder2));
timePoints = [0 0.5 1 2.5 3 3.5];


q0 = [114.5, 217.5, 160, 244.5, 142.5, 196];
shoulder1 = [180 240 110];
shoulder2 = [180 270 150];
shoulder3 = [180 240 140];
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220];

timePoints = [0 1 2 3.5 5.5];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6]*pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end),1,2)];
% wayPoints = [q0.', q0.', wayPoints]*pi/180;
% wayPoints(wayPoints < 0) = wayPoints(wayPoints < 0) + 360;
tvec = 0:0.01:timePoints(end);
N = length(tvec);
[qd,dqd,ddqd,pp] = cubicpolytraj(wayPoints,timePoints,tvec);

save('qd.mat','qd');
save('dqd.mat','dqd');
save('ddqd.mat','ddqd');
save('tvec.mat','tvec');