import numpy as np

def channel_exchange(points):
    x,y,z = points[:,0].copy(),points[:,1].copy(),points[:,2].copy()
    return np.stack([z,x,y],axis=1)

def channel_back(points):
    z,x,y = points[:,0].copy(),points[:,1].copy(),points[:,2].copy()
    return np.stack([x,y,z],axis=1)

def rigid_transform(points,T):
    pnum = points.shape[0]
    homo_points = np.concatenate([points,np.ones((pnum,1))],axis=1)
    return (T @ homo_points.T).T[:,:3]

def K(intrinsic):
    intrinsic = intrinsic.split()[1::2]
    f_u = float(intrinsic[0])
    f_v = float(intrinsic[1])
    cu = float(intrinsic[2])
    cv = float(intrinsic[3])
    return np.array([[f,0.0,cu],[0,f,cv],[0,0,1]])

def E(extrinsic):
    extrinsic = extrinsic.split()[1::2]
    extrinsic = [float(e) for e in extrinsic]
    return np.array(extrinsic).reshape((4,4))


def disp_to_rect(disp,scale_factor):
    """
    :param disp: the disparity matrix of the image
    :param scale_factor: the scale factor to recover the depth
    """

    H,W = disp.shape
    x,y = np.meshgrid(range(W),range(H))
    y = H - y
    depth = scale_factor/disp
    rect = np.stack([x,y,depth],axis=2)
    rect = rect.reshape(-1,3).astype(np.float32)
    return rect

def rect_to_vole(points,K):
    cu = K[0,2]
    cv = K[1,2]
    f_u = K[0,0]
    f_v = K[1,1]
    points[:,0] = (points[:,0]-cu)*points[:,2]/f_u
    points[:,1] = (points[:,1]-cv)*points[:,2]/f_v
    return points

def pto_ang_map(velo_points, H=64, W=512, slice=1):
    """
    :param H: the row num of depth map, could be 64(default), 32, 16
    :param W: the col num of depth map
    :param slice: output every slice lines
    """

    dtheta = np.radians(0.4 * 64.0 / H)
    dphi = np.radians(90.0 / W)

    x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)
    d[d == 0] = 0.000001
    r[r == 0] = 0.000001
    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= W] = W - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= H] = H - 1

    depth_map = - np.ones((H, W, 3))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map = depth_map[0::slice, :, :]
    depth_map = depth_map.reshape((-1, 3))
    depth_map = depth_map[depth_map[:, 0] != -1.0]
    return depth_map


def gen_sparse_points(pc_velo,H=64,W=512,slice=1):

    # depth, width, height
    valid_inds = (pc_velo[:, 0] < 120) & \
                 (pc_velo[:, 0] >= 0) & \
                 (pc_velo[:, 1] < 80) & \
                 (pc_velo[:, 1] >= -80) & \
                 (pc_velo[:, 2] < 1.5) & \
                 (pc_velo[:, 2] >= -2.5)
    pc_velo = pc_velo[valid_inds]
    return pto_ang_map(pc_velo, H, W, slice)

def project_disp_to_depth(calib, depth, max_high):
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows))
    points = np.stack([c, r, depth])
    points = points.reshape((3, -1))
    points = points.T
    cloud = calib.project_image_to_velo(points)
    valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
    return cloud[valid]

def merge(colors,scale_factor):
    K,E = camera_matrix()
    mpoints = []
    for i in range(5):
        disp = read_file('%d.npy'%(i+1))
        points = disp_to_rect(disp,scale_factor)
        points = rect_to_vole(points,K[i])
        points = channel_exchange(points)
        points = gen_sparse_points(points)
        #points = channel_back(points)
        points = rigid_transform(points,E[i])
        points = colorize(points,colors[i])
        mpoints.append(points)
    return np.concatenate(mpoints,axis=0),[p[:,:3] for p in mpoints]

def single(idx):
    K,E = camera_matrix()
    disp = read_file('%d.npy'%idx)
    points = disp_to_rect(disp,1.)
    points = rect_to_vole(points,K[0])
    points = channel_exchange(points)
    points = gen_sparse_points(points)
    return points



# def camera_matrix():
#     #FRONT
#     K1 = K("intrinsic: 2055.556149361639\
#             intrinsic: 2055.556149361639\
#             intrinsic: 939.6574698861468\
#             intrinsic: 641.0721821943271\
#             intrinsic: 0.03231600849798887\
#             intrinsic: -0.3214124825527059\
#             intrinsic: 0.0007932583953709973\
#             intrinsic: -0.0006257493541333847\
#             intrinsic: 0.0")
#     E1 = E("transform: 0.9998926849887427\
#             transform: -0.005993208400016058\
#             transform: 0.0133678704017097\
#             transform: 1.5389142447125008\
#             transform: 0.006042236521329663\
#             transform: 0.9999751560547995\
#             transform: -0.003630241176497072\
#             transform: -0.02363394083934774\
#             transform: -0.013345781499156929\
#             transform: 0.003710623431877962\
#             transform: 0.999904056092345\
#             transform: 2.115270572975561\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 1.0")
#     #FRONT LEFT
#     K2 = K("intrinsic: 2063.7008688972\
#             intrinsic: 2063.7008688972\
#             intrinsic: 970.7315379934879\
#             intrinsic: 639.9082229848484\
#             intrinsic: 0.03119623557580319\
#             intrinsic: -0.34029064830905453\
#             intrinsic: -0.0006801050887136624\
#             intrinsic: 0.001067963528920262\
#             intrinsic: 0.0")
#     E2 = E("transform: 0.7163508489464225\
#             transform: -0.6976495294008019\
#             transform: 0.011251459486630241\
#             transform: 1.492930189258495\
#             transform: 0.6976096514642995\
#             transform: 0.7164356249377603\
#             transform: 0.007795479709391459\
#             transform: 0.09192224912318936\
#             transform: -0.01349945915947628\
#             transform: 0.0022648282231656253\
#             transform: 0.9999063131891514\
#             transform: 2.1152105284507554\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 1.0")
#     #FRONT RIGHT
#     K4 = K("intrinsic: 2056.0892196793116\
#             intrinsic: 2056.0892196793116\
#             intrinsic: 935.743715862858\
#             intrinsic: 624.4064324983569\
#             intrinsic: 0.03490672761153742\
#             intrinsic: -0.3141180156718857\
#             intrinsic: 0.0012619308568439455\
#             intrinsic: -0.0026996059700096116\
#             intrinsic: 0.0")
#     E4 = E("transform: 0.716582596758835\
#             transform: 0.6975019454936628\
#             transform: -0.000646571666883216\
#             transform: 1.490324906589904\
#             transform: -0.6974986018023605\
#             transform: 0.71657554968999\
#             transform: -0.0038964176163232226\
#             transform: -0.09385927001229258\
#             transform: -0.002254441420230414\
#             transform: 0.003243087887177829\
#             transform: 0.9999921999069986\
#             transform: 2.1154927516413125\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 1.0") 
#     #SIDE LEFT
#     K3 = K("intrinsic: 2066.834902319412\
#             intrinsic: 2066.834902319412\
#             intrinsic: 952.8608233319966\
#             intrinsic: 249.49859488407833\
#             intrinsic: 0.044941016139828975\
#             intrinsic: -0.3435919955955713\
#             intrinsic: 0.00013187735016933997\
#             intrinsic: -0.0011427074991115992\
#             intrinsic: 0.0")
#     E3 = E("transform: 0.0012964074354981448\
#             transform: -0.9999536359093453\
#             transform: 0.009541769198720481\
#             transform: 1.4314958432756546\
#             transform: 0.9997854749192748\
#             transform: 0.0014933173193322094\
#             transform: 0.020658512623705133\
#             transform: 0.11128578863910166\
#             transform: -0.020671803699754576\
#             transform: 0.009512940400541162\
#             transform: 0.9997410567225569\
#             transform: 2.115330824237742\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 1.0")
#     #SIDE RIGHT
#     K5 = K("intrinsic: 2054.797701053215\
#             intrinsic: 2054.797701053215\
#             intrinsic: 972.7435431831157\
#             intrinsic: 242.4987260630087\
#             intrinsic: 0.03337281654124105\
#             intrinsic: -0.313299913027001\
#             intrinsic: -0.0001886604822006906\
#             intrinsic: -0.0012474351513059548\
#             intrinsic: 0.0")
#     E5 = E("transform: -0.00192725729868321\
#             transform: 0.9999977777956348\
#             transform: -0.000854449001917361\
#             transform: 1.428371747589099\
#             transform: -0.9999943442124861\
#             transform: -0.0019296044058050511\
#             transform: -0.00275466329645372\
#             transform: -0.11149205023810704\
#             transform: -0.002756305923587554\
#             transform: 0.0008491352243918596\
#             transform: 0.9999958408648639\
#             transform: 2.1156692490324467\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 0.0\
#             transform: 1.0")
#     return [K1,K2,K3,K4,K5],[E1,E2,E3,E4,E5]