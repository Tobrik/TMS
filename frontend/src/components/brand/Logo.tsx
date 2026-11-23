import { Stethoscope } from "lucide-react";
import { cn } from "@/lib/utils";

export interface LogoProps {
  size?: number;
  withText?: boolean;
  className?: string;
}

export function Logo({ size = 32, withText = true, className }: LogoProps) {
  return (
    <span className={cn("inline-flex items-center gap-2.5", className)}>
      <span
        className="inline-flex shrink-0 items-center justify-center rounded-[10px] bg-primary text-primary-foreground"
        style={{ width: size, height: size }}
        aria-hidden="true"
      >
        <Stethoscope style={{ width: size * 0.56, height: size * 0.56 }} />
      </span>
      {withText ? (
        <span
          className="font-display font-bold tracking-tight text-foreground"
          style={{ fontSize: Math.round(size * 0.66), lineHeight: 1 }}
        >
          TMS
        </span>
      ) : (
        <span className="sr-only">TMS</span>
      )}
    </span>
  );
}
