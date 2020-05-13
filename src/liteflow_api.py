from liteflownet.run import Network
class LiteFlow():
	def __init__(self):

		self.netNetwork = Network().cuda().eval()
		#TODO: Change the way to load state dict.

	def run(self,first,second):
		tenFirst = torch.FloatTensor(numpy.ascontiguousarray(first.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
		tenSecond = torch.FloatTensor(numpy.ascontiguousarray(second.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
		assert(tenFirst.shape[1] == tenSecond.shape[1])
		assert(tenFirst.shape[2] == tenSecond.shape[2])

		intWidth = tenFirst.shape[2]
		intHeight = tenFirst.shape[1]

		assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
		assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

		tenPreprocessedFirst = tenFirst.cuda().view(1, 3, intHeight, intWidth)
		tenPreprocessedSecond = tenSecond.cuda().view(1, 3, intHeight, intWidth)

		intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
		intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

		tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
		tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

		tenFlow = torch.nn.functional.interpolate(input=self.netNetwork(tenPreprocessedFirst, tenPreprocessedSecond), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

		tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
		tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

		return tenFlow[0, :, :, :].cpu()
