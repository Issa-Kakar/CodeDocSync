"""
Production error handling for suggestion system.

Provides comprehensive error classes, graceful degradation strategies,
and error recovery mechanisms for the suggestion generation pipeline.
"""

import logging
from collections.abc import Callable
from typing import TypeVar

from .models import Suggestion, SuggestionContext, SuggestionType

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SuggestionError(Exception):
    """Base exception for suggestion system."""

    def __init__(
        self,
        message: str,
        context: SuggestionContext | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize suggestion error.

        Args:
            message: Error message
            context: Context when error occurred
            original_error: Original exception that caused this error
        """
        super().__init__(message)
        self.context = context
        self.original_error = original_error

    def __str__(self) -> str:
        """Return string representation."""
        parts = [super().__str__()]
        if self.context:
            parts.append(
                f" (Function: {self.context.function.signature.name} "
                f"at {self.context.function.file_path}:{self.context.function.line_number})"
            )
        if self.original_error:
            parts.append(
                f" [Caused by: {type(self.original_error).__name__}: {self.original_error}]"
            )
        return "".join(parts)


class StyleDetectionError(SuggestionError):
    """Failed to detect docstring style."""

    def __init__(
        self,
        message: str,
        fallback_style: str = "google",
        attempted_text: str | None = None,
        **kwargs,
    ):
        """
        Initialize style detection error.

        Args:
            message: Error message
            fallback_style: Style to use as fallback
            attempted_text: Text that failed detection
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)
        self.fallback_style = fallback_style
        self.attempted_text = attempted_text


class SuggestionGenerationError(SuggestionError):
    """Failed to generate suggestion."""

    def __init__(
        self,
        message: str,
        partial_result: str | None = None,
        suggestion_type: SuggestionType | None = None,
        **kwargs,
    ):
        """
        Initialize suggestion generation error.

        Args:
            message: Error message
            partial_result: Partial suggestion if available
            suggestion_type: Type of suggestion that failed
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)
        self.partial_result = partial_result
        self.suggestion_type = suggestion_type


class TemplateRenderError(SuggestionError):
    """Failed to render template."""

    def __init__(
        self,
        message: str,
        template_name: str,
        section: str | None = None,
        **kwargs,
    ):
        """
        Initialize template render error.

        Args:
            message: Error message
            template_name: Name of template that failed
            section: Section that failed to render
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)
        self.template_name = template_name
        self.section = section


class ValidationError(SuggestionError):
    """Suggestion validation failed."""

    def __init__(
        self,
        message: str,
        validation_type: str,
        invalid_content: str | None = None,
        **kwargs,
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            validation_type: Type of validation that failed
            invalid_content: Content that failed validation
            **kwargs: Additional arguments for base class
        """
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        self.invalid_content = invalid_content


def create_fallback_suggestion(
    context: SuggestionContext,
    error: Exception | None = None,
) -> Suggestion:
    """
    Create a fallback suggestion when generation fails.

    Args:
        context: Suggestion context
        error: Error that caused the fallback

    Returns:
        Basic suggestion with low confidence
    """
    # Create a basic suggestion based on issue type
    issue = context.issue
    fallback_text = issue.suggestion  # Use analyzer's basic suggestion

    # Add error context if available
    if error:
        fallback_text += f"\n\n# Note: Automated suggestion failed: {error}"

    return Suggestion(
        type=SuggestionType.FULL_DOCSTRING,
        original_text=context.docstring.raw_text if context.docstring else "",
        suggested_text=fallback_text,
        confidence=0.3,  # Low confidence for fallback
        style=context.project_style or "google",
        issue_id=f"{issue.issue_type}:{issue.line_number}",
        copy_paste_ready=False,  # Fallback suggestions need review
        metadata={
            "fallback": True,
            "error": str(error) if error else None,
        },
    )


def with_suggestion_fallback(
    fallback_func: Callable[[SuggestionContext], Suggestion] | None = None,
) -> Callable:
    """
    Decorator for graceful degradation in suggestion generation.

    Args:
        fallback_func: Optional custom fallback function

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except SuggestionGenerationError as e:
                # Log the error
                logger.warning(f"Suggestion generation failed: {e}")

                # Return partial result if available
                if e.partial_result and hasattr(e, "context"):
                    return Suggestion(
                        type=e.suggestion_type or SuggestionType.FULL_DOCSTRING,
                        original_text=(
                            e.context.docstring.raw_text if e.context.docstring else ""
                        ),
                        suggested_text=e.partial_result,
                        confidence=0.5,  # Medium confidence for partial
                        style=e.context.project_style or "google",
                        issue_id=f"{e.context.issue.issue_type}:{e.context.issue.line_number}",
                        copy_paste_ready=False,
                        metadata={"partial": True, "error": str(e)},
                    )

                # Use custom fallback if provided
                if fallback_func and len(args) > 0:
                    # Assume first arg is context
                    context = (
                        args[0] if isinstance(args[0], SuggestionContext) else None
                    )
                    if context:
                        return fallback_func(context)

                # Use default fallback
                if hasattr(e, "context") and e.context:
                    return create_fallback_suggestion(e.context, e)

                # Re-raise if no fallback possible
                raise

            except (StyleDetectionError, TemplateRenderError, ValidationError) as e:
                logger.error(f"Critical suggestion error: {e}")

                # Try to create minimal suggestion
                if hasattr(e, "context") and e.context:
                    return create_fallback_suggestion(e.context, e)

                # Re-raise if no context
                raise

            except Exception as e:
                # Unexpected error
                logger.exception(f"Unexpected error in suggestion generation: {e}")

                # Try to extract context from args
                context = None
                for arg in args:
                    if isinstance(arg, SuggestionContext):
                        context = arg
                        break

                if context:
                    return create_fallback_suggestion(context, e)

                # Re-raise if no recovery possible
                raise

        return wrapper

    return decorator


class ErrorRecoveryStrategy:
    """Strategy for recovering from errors during suggestion generation."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        fallback_styles: list[str] | None = None,
    ):
        """
        Initialize error recovery strategy.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            fallback_styles: List of fallback styles to try
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.fallback_styles = fallback_styles or ["google", "numpy", "sphinx"]

    def should_retry(self, error: Exception, attempt: int) -> bool:
        """
        Determine if operation should be retried.

        Args:
            error: Error that occurred
            attempt: Current attempt number

        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False

        # Retry on specific errors
        retriable_errors = (
            TemplateRenderError,
            ValidationError,
        )

        return isinstance(error, retriable_errors)

    def get_fallback_style(self, attempted_styles: list[str]) -> str | None:
        """
        Get next fallback style to try.

        Args:
            attempted_styles: Styles already attempted

        Returns:
            Next style to try, or None
        """
        for style in self.fallback_styles:
            if style not in attempted_styles:
                return style
        return None


class SuggestionErrorHandler:
    """Centralized error handling for suggestion system."""

    def __init__(self, recovery_strategy: ErrorRecoveryStrategy | None = None):
        """
        Initialize error handler.

        Args:
            recovery_strategy: Strategy for error recovery
        """
        self.recovery_strategy = recovery_strategy or ErrorRecoveryStrategy()
        self.error_counts: dict[str, int] = {}

    def handle_error(
        self,
        error: Exception,
        context: SuggestionContext | None = None,
    ) -> Suggestion | None:
        """
        Handle an error and attempt recovery.

        Args:
            error: Error that occurred
            context: Context when error occurred

        Returns:
            Recovery suggestion if possible, None otherwise
        """
        # Track error frequency
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Log appropriately based on frequency
        if self.error_counts[error_type] > 10:
            logger.debug(f"Frequent error: {error_type}: {error}")
        else:
            logger.warning(f"Handling error: {error_type}: {error}")

        # Attempt recovery based on error type
        if isinstance(error, StyleDetectionError):
            # Use fallback style
            if context:
                context.project_style = error.fallback_style
                logger.info(f"Using fallback style: {error.fallback_style}")
                return None  # Let caller retry with new style

        elif isinstance(error, SuggestionGenerationError):
            # Use partial result if available
            if error.partial_result:
                return Suggestion(
                    type=error.suggestion_type or SuggestionType.FULL_DOCSTRING,
                    original_text=(
                        context.docstring.raw_text
                        if context and context.docstring
                        else ""
                    ),
                    suggested_text=error.partial_result,
                    confidence=0.4,
                    style=context.project_style if context else "google",
                    issue_id=(
                        f"{context.issue.issue_type}:{context.issue.line_number}"
                        if context
                        else "unknown"
                    ),
                    copy_paste_ready=False,
                    metadata={"error_recovery": True},
                )

        elif isinstance(error, ValidationError):
            # Create minimal valid suggestion
            if context:
                return create_fallback_suggestion(context, error)

        # Default: create fallback if context available
        if context:
            return create_fallback_suggestion(context, error)

        return None

    def reset_error_counts(self) -> None:
        """Reset error frequency counts."""
        self.error_counts.clear()


# Global error handler instance
_error_handler = SuggestionErrorHandler()


def get_error_handler() -> SuggestionErrorHandler:
    """Get global error handler instance."""
    return _error_handler


def handle_suggestion_error(
    error: Exception,
    context: SuggestionContext | None = None,
) -> Suggestion | None:
    """
    Handle a suggestion error using global handler.

    Args:
        error: Error that occurred
        context: Context when error occurred

    Returns:
        Recovery suggestion if possible, None otherwise
    """
    return _error_handler.handle_error(error, context)
