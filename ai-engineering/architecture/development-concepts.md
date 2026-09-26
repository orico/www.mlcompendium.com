# Development Concepts

ML code still has to be software, and software has names for the shapes that keep it maintainable. The page starts with design patterns, the catalogue of those shapes, and then moves to dependency injection and SOLID, the object-oriented rules underneath them.

## Patterns

Before any rule, it helps to see the patterns side by side. [Software and architectural design patterns](https://github.com/DovAmir/awesome-design-patterns) is DovAmir's curated list of software and architecture related design patterns, and the figure below is taken from it.

<figure><img src="../../ops/.gitbook/assets/gimg-d9bb97d36ff5.png" alt=""><figcaption><p>DovAmir</p><p>Credit: <a href="https://lh4.googleusercontent.com/qgvN9nWhe0NRVvcvILrJsF2UeAqZ4H8CIcAUWOBMsXlFEAxhvNCnfQiFrwtLgiXaN1DiziRZ-cjefQzwaBWjtpE3q5SDRlZ9m6-sdJ0NtFnCs4CB4ZSk9Ay9G9X0U6Gy6cLy8_nM">copied from the original hosted image.</a></p></figcaption></figure>

## Programming concepts

With the patterns named, the next question is why they are built the way they are, and dependency injection is the clearest example. [Dependency injection](https://www.freecodecamp.org/news/a-quick-intro-to-dependency-injection-what-it-is-and-when-to-use-it-7578c84fa88f/) is Bhavya Karia's quick intro: one object supplies the dependencies, the services, of another. It is based on [SOLID](https://scotch.io/bar-talk/s-o-l-i-d-the-first-five-principles-of-object-oriented-design#toc-single-responsibility-principle): the class should do one thing, so we are letting other classes create 3rd party/class objects for us instead of doing it internally, either by init passing or by injecting in runtime.

That single-responsibility rule is only the first letter. [SOLID](https://scotch.io/bar-talk/s-o-l-i-d-the-first-five-principles-of-object-oriented-design#toc-single-responsibility-principle) — the five principles of object oriented design, explained as the way to write cleaner, scalable, and maintainable code.
