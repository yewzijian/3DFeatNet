algos = {'f3d_d32'};
legends = {'3dfeat-net'};

distances = 0.1 : 0.1 : 10;

numAlgos = length(algos);
precisions = {};
algoNames = {};

for iAlgo = 1 : numAlgos
    algoName = algos{iAlgo};
    load(fullfile('results_oxford', sprintf('matching_statistic-%s.mat', algoName)));
    
    distTable = cat(1, statisticTable.nearestMatchDist{:});
    
    precisionTable = zeros(size(distances));
    for iDist = 1 : length(distances)
        precisionTable(iDist) = nnz(distTable < distances(iDist));
    end
    precisionTable = precisionTable/length(distTable);
    
    precisions{iAlgo} = precisionTable * 100;
    algoNames{iAlgo} = algoName;
end

%%
figure(1), clf, hold on
for iAlgo = 1 : numAlgos
    if iAlgo > 4
        plot(distances, precisions{iAlgo}, '--');
    else
        plot(distances, precisions{iAlgo});
    end
end

ylim([0 55])
yl = ylim;
plot([1, 1], yl, '--k');

% legend(legends, 'Location', 'Northwest')
legend(legends, 'Location', 'EastOutside')
ylabel('Precision (%)');
xlabel('Meters') 
