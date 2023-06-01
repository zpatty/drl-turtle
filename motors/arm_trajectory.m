%% 3D Arm Trajectory
d = 0.04337;
theta = [0 0.4 0.1 -0.2;
        0 0.0 0.0 0.0;
        0 1 2 2];
phi = [0 pi/6 pi/4 pi/3;
        0 0 0 0;
        0 -pi/4 -pi/4 -pi/4];
dL = [-0.0 -0.02 -0.05 -0.05;
    -0.0 -0.065 -0.065 -0.065;
    -0.0 -0.0 -0.0 -0.0];

dx = theta.*d.*cos(phi);
dy = theta.*d.*sin(phi);

motor = [0 0 pi/4 pi/3;
        0 -0.0 -pi/4 -pi/4;
        0 pi/4 pi/3 0.0];

wayPoints = [motor(1,:);dx(1,:);dy(1,:);dL(1,:);motor(2,:);dx(2,:);dy(2,:);dL(2,:);motor(3,:);
    dx(3,:);dy(3,:);dL(3,:)];
timePoints = [0 3 6 9];
tvec = 0:0.05:9;
N = length(tSamples);
[qd,dqd,ddqd,pp] = cubicpolytraj(wayPoints,timePoints,tSamples);

save("~/SRRA-three_mod/qd.mat", "qd")
save("~/SRRA-three_mod/dqd.mat", "dqd")
save("~/SRRA-three_mod/ddqd.mat", "ddqd")
save("~/SRRA-three_mod/tvec.mat", "tvec")



