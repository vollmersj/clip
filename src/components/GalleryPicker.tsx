import { data, galleryUrl } from '../lib/data';

interface GalleryPickerProps {
  selected: number;
  onSelect: (index: number) => void;
  showLabels?: boolean;
}

/** Thumbnail strip for choosing one of the 8 gallery images. */
export function GalleryPicker({ selected, onSelect, showLabels = true }: GalleryPickerProps) {
  return (
    <div className="gallery-strip">
      {data.items.map((item, i) => (
        <button
          key={item.id}
          className={`gallery-pick${i === selected ? ' selected' : ''}`}
          onClick={() => onSelect(i)}
          title={item.caption}
        >
          <img className="thumb" src={galleryUrl(item.file)} alt={item.caption} />
          {showLabels && <span className="pick-label">{item.className}</span>}
        </button>
      ))}
    </div>
  );
}
