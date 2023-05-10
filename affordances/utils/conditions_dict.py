IK=False
conditions = { 
                'Random': 
                    {
                        'init_learner': 'random',
                        'optimal_ik':IK,
                        'segment':False
                    },
                'Binary':
                    {
                        'init_learner': 'binary',
                        'optimal_ik':IK,
                        'segment':False
                    }, 
                'GVF':
                    {
                        'init_learner': 'gvf',
                        'optimal_ik':IK,
                        'segment':False
                    },
                'Weighted':
                    {
                        'init_learner': 'weighted-binary',
                        'optimal_ik':IK,
                        'segment':False
                    },   
                 # 'Random-soft': 
                 #    {
                 #        'init_learner': 'random',
                 #        'optimal_ik':IK,
                 #        'segment':False,
                 #        'sampler':'soft'
                 #    },
                # 'Binary-soft':
                #     {
                #         'init_learner': 'binary',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'soft'
                #     }, 
                # 'GVF-soft':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'soft'
                #     },
                # 'Weighted-soft':
                #     {
                #         'init_learner': 'weighted-binary',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'soft'
                #     }, 
                # 'Random-Sum': 
                #     {
                #         'init_learner': 'random',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'sum'
                #     },
                # 'Random': 
                #     {
                #         'init_learner': 'random',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         # 'sampler':'max'
                #     },
                # 'Binary-Sum':
                #     {
                #         'init_learner': 'binary',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'sum'
                #     }, 
                # 'GVF-Sum':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'sum',
                #         'uncertainty':'none'
                #     },
                # 'Weighted-Sum':
                #     {
                #         'init_learner': 'weighted-binary',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'sum'
                #     }, 

                # 'Binary-Max':
                #     {
                #         'init_learner': 'binary',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'max'
                #     }, 
                # 'GVF-Max':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'max',
                #         'uncertainty':'none,'

                #     },
                # 'Weighted-Max':
                #     {
                #         'init_learner': 'weighted-binary',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'max'
                #     }, 
                # 'GVF-Sum-Bonus':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'sum',
                #         'uncertainty':'bonus'
                #     },
                # 'GVF-Max-Bonus':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':IK,
                #         'segment':False,
                #         'sampler':'max',
                #         'uncertainty':'bonus'
                #     },

                  
                #  'Random-Segmented': 
                #     {
                #         'init_learner': 'random',
                #         'optimal_ik':IK,
                #         'segment':True
                #     },
                # 'Binary-Segmented':
                #     {
                #         'init_learner': 'binary',
                #         'optimal_ik':IK,
                #         'segment':True
                #     }, 
                # 'GVF-Segmented':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':IK,
                #         'segment':True
                #     },
                # 'Weighted-Segmented':
                #     {
                #         'init_learner': 'weighted-binary',
                #         'optimal_ik':IK,
                #         'segment':True
                #     },  
                # 'Random-Optimal': 
                #     {
                #         'init_learner': 'random',
                #         'optimal_ik':True,
                #         'segment':False
                #     },
                # 'Binary-Optimal':
                #     {
                #         'init_learner': 'binary',
                #         'optimal_ik':True,
                #         'segment':False
                #     }, 
                # 'GVF-Optimal':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':True,
                #         'segment':False
                #     },
                # 'Weighted-Optimal':
                #     {
                #         'init_learner': 'weighted-binary',
                #         'optimal_ik':True,
                #         'segment':False
                #     },   
                # 'Random-Optimal-Seg': 
                #     {
                #         'init_learner': 'random',
                #         'optimal_ik':True,
                #         'segment':True
                #     },
                # 'Binary-Optimal-Seg':
                #     {
                #         'init_learner': 'binary',
                #         'optimal_ik':True,
                #         'segment':True
                #     }, 
                # 'GVF-Optimal-Seg':
                #     {
                #         'init_learner': 'gvf',
                #         'optimal_ik':True,
                #         'segment':True
                #     },
                # 'Weighted-Optimal-Seg':
                #     {
                #         'init_learner': 'weighted-binary',
                #         'optimal_ik':True,
                #         'segment':True
                #     },   
                # 'Oracle': 
                #     {
                #         'init_learner': 'random',
                #         'optimal_ik':True,
                #         'segment':True
                #     },    
   

}