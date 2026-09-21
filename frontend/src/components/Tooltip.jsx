import * as TooltipPrimitive from '@radix-ui/react-tooltip'

/** Wrap the app once in TooltipProvider; each Tooltip is then just a Root. */
export const TooltipProvider = TooltipPrimitive.Provider

export function Tooltip({ children, text }) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          className="rounded bg-gray-900 px-2 py-1 text-xs text-white shadow-lg max-w-xs"
          sideOffset={5}
        >
          {text}
          <TooltipPrimitive.Arrow className="fill-gray-900" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}
