class Videoer:
    """
    Obtains video id from instances or e2e dataset.

    It gets video id from identities:
    - If the identity id contains dataset0, dataset1, dataset2 then the identity is returned
    - If the identity id contains dataset3, then the video id is extracted e.g. dataset3_004-005_1 -> dataset3_004-005
    """

    def __init__(self, ds, identity_key="identity_id"):
        self.ds = ds
        self.identity_key = identity_key

        self.video2index = {}
        self.index2video = {}

        for i in range(len(self.ds)):
            id = self.ds.id(i)
            tgt_ann = self.ds.tgt_ann(id)

            identity_id = tgt_ann[self.identity_key]

            if (
                "dataset0" in identity_id
                or "dataset1" in identity_id
                or "dataset2" in identity_id
            ):
                video_id = identity_id

            elif "dataset3" in identity_id:
                # e.g. dataset3_004-005_1, drop _1 and take the others
                video_id = "_".join(identity_id.split("_")[:-1])

            else:
                # legacy, i.e. realcolon only (v_id)
                video_id = identity_id.split("_")[0]

            if video_id not in self.video2index:
                self.video2index[video_id] = []

            self.video2index[video_id].append(id)
            self.index2video[id] = video_id

    def get_video_id(self, instance_id):
        return self.index2video[instance_id]

    def get_instances(self, video_id):
        return self.video2index[video_id]

    def list_video_ids(self):
        return list(self.video2index.keys())
