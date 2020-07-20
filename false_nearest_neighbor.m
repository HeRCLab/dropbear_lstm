function [metric1,metric2] = false_nearest_neighbor (d,d_plus_one,v_s)
    dist1 = norm(d(1,:)-d(2,:));
    dist2 = norm(d_plus_one(1,:)-d_plus_one(2,:));
    metric1 = (dist1-dist2)/dist1;
    metric2 = dist2/v_s;
end
