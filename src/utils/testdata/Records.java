package demo.records;

public sealed interface Shape permits Circle, Rectangle {}

public record Circle(double radius) implements Shape {}

public record Rectangle(double width, double height) implements Shape {}
