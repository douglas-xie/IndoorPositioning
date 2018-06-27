function [output] = calDistance(NumberOfTag, NumberOfReader, PosTag, PosReader)
distance = zeros(NumberOfTag, NumberOfReader);
for j = 1:NumberOfTag
    for i = 1:NumberOfReader
        distance(j,i) = norm(PosTag(j, :) - PosReader(i, :));
    end
end
output = distance;
end