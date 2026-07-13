interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  disabled?: boolean;
}

export function Slider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => String(v),
  disabled,
}: SliderProps) {
  return (
    <label className="control">
      <span className="control-label">
        <span>{label}</span>
        <span className="control-value">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
