from timelens.common import event, image_sequence, iterator_modifiers

class HybridStorage(object):
    """Template class that stores events and images."""

    def __init__(self, images, events):
        # 图像数据对象(图像.png、时间戳)和事件数据对象(事件.npz、图像尺寸)
        self._images = images
        self._events = events

    def get_image_size(self):
        return self._images._height, self._images._width

    def make_interframe_events_iterator(self, number_of_skips):
        # 这里就是image的时间戳信息,间隔是number_of_skip
        timestamps = list(self.make_boundary_timestamps_iterator(number_of_skips))
        return self._events.make_sequential_iterator(timestamps)

    def make_boundary_timestamps_iterator(self, number_of_skips):
        return iterator_modifiers.make_skip_iterator(
            iter(self._images._timestamps), number_of_skips
        )

    def make_pair_boundary_timestamps_iterator(self, number_of_skips):
        return iterator_modifiers.make_iterator_over_groups(
            iterator_modifiers.make_skip_iterator(
                iter(self._images._timestamps), number_of_skips
            ),
            2,
        )

    def make_boundary_frames_iterator(self, number_of_skips):
        return iterator_modifiers.make_iterator_over_groups(
            iterator_modifiers.make_skip_iterator(
                iter(self._images._images), number_of_skips
            ),
            2,
        )

    @classmethod
    def from_folders_jit(
            cls,
            event_folder,
            image_folder,
            event_file_template="{:06d}.npz",
            image_file_template="{:06d}.png",
            cropping_data=None,
            timestamps_file="timestamp.txt"
    ):
        images = image_sequence.ImageSequence.from_folder(
            folder=image_folder,
            image_file_template=image_file_template,
            timestamps_file=timestamps_file
        )
        events = event.EventJITSequence.from_folder(
            folder=event_folder,
            image_height=images._height,
            image_width=images._width,
            event_file_template=event_file_template
        )

        return cls(images, events)

    @classmethod
    def from_folders(
            cls,
            event_folder,
            image_folder,
            event_file_template="{:06d}.npz",
            image_file_template="{:06d}.png",
            cropping_data=None,
            timestamps_file="timestamp.txt"
    ):
        # 这里的images和events到底指啥呢
        # 这两个也是类似的方式，返回了cls()对象，就相当于是构造了呗
        # 返回一个ImageSequence对象, 其中有images和timestamps属性
        images = image_sequence.ImageSequence.from_folder(
            folder=image_folder,
            image_file_template=image_file_template,
            timestamps_file=timestamps_file
        )
        # 返回一个EventSequence对象, 其中有events信息,图像尺寸，开始结束时间
        events = event.EventSequence.from_folder(
            folder=event_folder,
            image_height=images._height,
            image_width=images._width,
            event_file_template=event_file_template
        )

        return cls(images, events)
