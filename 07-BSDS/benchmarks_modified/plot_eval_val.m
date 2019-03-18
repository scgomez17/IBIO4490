function plot_eval_val(evalDir,evalDir2,evalDir3,evalDir4)
% plot evaluation results.
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>
% This function was modified in order to save the image with and specific
% name and specific title. This modification also can recieve 2 folders to make a subplot
open('isoF.fig');
hold on
for i=1:4
    if i==1
        folder= evalDir;
        col='r';
    elseif i==2
        folder= evalDir2;
        col='b';
    elseif i==3
        folder= evalDir3;
        col='k';
    elseif i==4
        folder= evalDir4;
        col='m';
    end
      
    fwrite(2,sprintf('\n%s\n',folder));
    
    if exist(fullfile(folder,'eval_bdry_thr.txt'),'file'),
        
        prvals = dlmread(fullfile(folder,'eval_bdry_thr.txt')); % thresh,r,p,f
        f=find(prvals(:,2)>=0.01);
        prvals = prvals(f,:);
        
        evalRes = dlmread(fullfile(folder,'eval_bdry.txt'));
        
        if size(prvals,1)>1,
            plot(prvals(1:end,2),prvals(1:end,3),col,'LineWidth',3);
            title ('Precision vs Recall for val set')
        else
            plot(evalRes(2),evalRes(3),'o','MarkerFaceColor',col,'MarkerEdgeColor',col,'MarkerSize',8);
            title ('Precision vs Recall for val set')
        end
        
        if i==4
            hold off
            %legend('human','curves','watershed','kmeans')
            h= findobj('Color','g');
            set(h, 'DisplayName','Human')
            w= findobj('Color','r');
            set(w, 'DisplayName','Watershed')
            wf= findobj('Color','b');
            set(wf, 'DisplayName','Watershed-filter')
            k= findobj('Color','k');
            set(k, 'DisplayName','K-means')
            kf= findobj('Color','m');
            set(kf, 'DisplayName','K-means-filter')
            L= [h(1) w(1) wf(1) k(1) kf(1)];            
            legend(L)
            name= 'PR_val.fig';
            savefig(name)
        end
        
        fprintf('Boundary\n');
        fprintf('ODS: F( %1.2f, %1.2f ) = %1.2f   [th = %1.2f]\n',evalRes(2:4),evalRes(1));
        fprintf('OIS: F( %1.2f, %1.2f ) = %1.2f\n',evalRes(5:7));
        fprintf('Area_PR = %1.2f\n\n',evalRes(8));
    end
    
    if exist(fullfile(evalDir,'eval_cover.txt'),'file'),
        evalRes = dlmread(fullfile(evalDir,'eval_cover.txt'));
        fprintf('Region\n');
        fprintf('GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n',evalRes(2),evalRes(1),evalRes(3:4));
        evalRes = dlmread(fullfile(evalDir,'eval_RI_VOI.txt'));
        fprintf('Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes(2),evalRes(1),evalRes(3));
        fprintf('Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n',evalRes(5),evalRes(4),evalRes(6));
        
    end
end