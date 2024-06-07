clear
up = 240;
down = 110;

straight = [230 150 230];
turn = 30;
% front fins
shoulder1 = [up down up];
shoulder2 = [straight];
shoulder3 = [240 140 240];
shoulder4 = 180 + (180 - shoulder1 - turn);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [120 220 120];

% back fins 
rear_yaw = 180;
rear_pitch = 180;
shoulder7 = [rear_pitch rear_pitch rear_pitch];
shoulder8 = [rear_yaw rear_yaw rear_yaw];
shoulder9 = 180 + (180 - shoulder8);
shoulder10 = 180 + (180 - shoulder7);
% time points for a single cycle
timePoints = [0 1 2];
% tshift = timePoints(2:end) + timePoints(end);
% timePoints_cycle = [timePoints, tshift(2:end)];
timePoints_cycle = timePoints;
wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
% wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
% tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
t_0 = second(datetime);
loop = 1;
i = 1;
folders = {'straight', 'turnlf', 'surface', 'dive'};
while loop < 5
    % if loop == 1
    n = get_qindex(second(datetime) - t_0, tvec);
    % else
    %     % print("done with first loop")
    %     offset = t_0 - 2;
    %     n = get_qindex(second(datetime) - offset, tvec);
    % end
    qd_now(:,i) = qd(:,n);
    if n > length(tvec) - 2
        % print(f"time: {(time.time() - offset)}\n")
        files = dir(strcat(folders{loop}, '/*.mat'));
        disp(folders{loop})
        for j=1:length(files)
            load(strcat(folders{loop},strcat('/', files(j).name)));
        end
        loop = loop + 1;
        t_0 = second(datetime);
    end
    i = i+1;
end

plot(1:length(qd_now),qd_now(1:6,:))


function qindex = get_qindex(mod_clock, tvec)
    % input: takes in the current time we're getting and makes sure we stay at proper
    % time element 
    % output: outputs the proper time vector index we need for our pos, vel, and accel vectors
    % Notes:
    % - recall tvec is a numpy matrix object that is (1,56) dim 
    qindex = 1;
    % mod_clock = seconds(mod_clock);
    for t = 2:length(tvec)
        if mod_clock >= tvec(t-1) && mod_clock < tvec(t)
            qindex = t - 1;
        elseif t==(length(tvec) - 1) && mod_clock >= tvec(t)
            qindex = t;
        end
    end
end
